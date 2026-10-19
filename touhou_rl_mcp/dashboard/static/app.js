/* 训练看板前端: 原生 JS + canvas, 无外部依赖。
 *
 * 设计取向:
 *   - 指标列表由后端从 train.jsonl 里**自动推断**, 这里不做任何字段假设,
 *     新增训练脚本/新指标不用改前端。
 *   - 一切都从 /api/run 的增量接口取数, 所以"训练中"和"训练后"是同一套代码。
 */
(() => {
  'use strict';

  const PALETTE = [
    '#5b8cff', '#4ade80', '#fbbf24', '#f472b6',
    '#22d3ee', '#a78bfa', '#fb923c', '#94a3b8',
  ];

  const $ = (id) => document.getElementById(id);

  const state = {
    runs: [],
    runId: null,
    episodes: [],
    metrics: [],
    selected: new Set(),
    actions: {},
    evalRows: [],
    smooth: 0,
    norm: false,
    live: false,
    timer: null,
    hoverX: null,
    layout: null,
    filter: '',
    showAll: false,
  };

  /* ------------------------------------------------------------------ */
  /* 工具                                                                */
  /* ------------------------------------------------------------------ */

  // 支持 ?run=<id> 深链接: 刷新 / 分享都能停在同一次训练上
  function syncUrl(id) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('run', id);
    else url.searchParams.delete('run');
    window.history.replaceState(null, '', url);
  }

  async function api(path) {
    const resp = await fetch(path, { cache: 'no-store' });
    if (!resp.ok) {
      let msg = resp.statusText;
      try { msg = (await resp.json()).error || msg; } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    return resp.json();
  }

  function dig(row, path) {
    return path.split('.').reduce((acc, key) => (acc == null ? acc : acc[key]), row);
  }

  function fmt(value) {
    if (!Number.isFinite(value)) return '—';
    const abs = Math.abs(value);
    if (abs >= 10000) return value.toExponential(1);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }

  function ago(seconds) {
    if (seconds < 90) return '刚刚';
    if (seconds < 3600) return `${Math.round(seconds / 60)} 分钟前`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)} 小时前`;
    return `${Math.round(seconds / 86400)} 天前`;
  }

  function setStatus(text) { $('status').textContent = text; }

  function color_of(metric) {
    const index = state.metrics.indexOf(metric);
    return PALETTE[(index < 0 ? 0 : index) % PALETTE.length];
  }

  /* ------------------------------------------------------------------ */
  /* 左侧 run 列表                                                       */
  /* ------------------------------------------------------------------ */
  async function loadRuns() {
    try {
      const data = await api('/api/runs');
      state.runs = data.runs || [];
      renderRuns();
      const shown = state.runs.filter((run) => state.showAll || !run.hidden).length;
      setStatus(shown === state.runs.length
        ? `${shown} 个 run`
        : `${shown} / ${state.runs.length} 个 run`);
    } catch (err) {
      setStatus('扫描失败: ' + err.message);
    }
  }

  function renderRuns() {
    const box = $('runs');
    const needle = state.filter.trim().toLowerCase();
    const list = state.runs.filter((run) => {
      if (run.hidden && !state.showAll) return false;
      return !needle || run.id.toLowerCase().includes(needle) || run.name.toLowerCase().includes(needle);
    });
    box.innerHTML = '';
    if (!list.length) {
      box.innerHTML = '<div class="empty small" style="padding:12px">没有匹配的 run</div>';
      return;
    }
    for (const run of list) {
      const el = document.createElement('div');
      el.className = 'run' + (run.id === state.runId ? ' active' : '');
      const live = run.age_s < 120 ? '<span class="pill live">live</span>' : '';
      const episodes = run.episodes ? `${run.episodes} 局` : '0 局';
      el.innerHTML = `
        <div class="r-name"><span>${escapeHtml(run.name)}</span>${live}</div>
        <div class="r-meta">
          <span>${episodes}</span>
          <span>${ago(run.age_s)}</span>
          ${run.has_eval ? '<span class="pill">eval</span>' : ''}
        </div>`;
      el.title = run.path;
      el.onclick = () => openRun(run.id);
      box.appendChild(el);
    }
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  /* ------------------------------------------------------------------ */
  /* 打开一个 run                                                        */
  /* ------------------------------------------------------------------ */
  async function openRun(id) {
    stopLive();
    state.runId = id;
    syncUrl(id);
    state.episodes = [];
    state.metrics = [];
    state.selected = new Set();
    state.actions = {};
    state.evalRows = [];
    state.hoverX = null;
    renderRuns();
    $('run-name').textContent = '载入中…';

    try {
      const data = await api(`/api/run?id=${encodeURIComponent(id)}&since=0`);
      applyPayload(data, { replace: true });
      const info = data.run || {};
      $('run-name').textContent = info.name || id;
      $('run-path').textContent = info.path || '';
      setStatus(`已载入 ${state.episodes.length} 局`);
    } catch (err) {
      $('run-name').textContent = '载入失败';
      $('run-path').textContent = err.message;
      setStatus('载入失败');
      return;
    }
    if (state.live) startLive();
  }

  function applyPayload(data, { replace = false } = {}) {
    const incoming = data.episodes || [];
    state.episodes = replace ? incoming : state.episodes.concat(incoming);
    state.actions = data.actions || state.actions;
    if (Array.isArray(data.info?.eval)) state.evalRows = data.info.eval;
    if (Array.isArray(data.metrics) && data.metrics.length) {
      state.metrics = data.metrics;
    }
    if (replace) {
      pickDefaultMetrics();
    } else {
      // 新指标出现时补上勾选 (例如训练中途才开始记录 eval)
      for (const metric of state.metrics) {
        if (!state.__known?.has(metric)) state.selected.add(metric);
      }
    }
    state.__known = new Set(state.metrics);
    renderMetrics();
    renderActions();
    renderEval();
    renderTable();
    draw();
  }

  function pickDefaultMetrics() {
    // 默认只选两条: "训练回报" 和 "泛化分数" —— 它们是决策依据, 且量纲接近, 不用归一化
    const preferred = ['return', 'eval', 'max_stage', 'frames', 'steps', 'loss'];
    const chosen = preferred.filter((m) => state.metrics.includes(m)).slice(0, 2);
    state.selected = new Set(chosen.length ? chosen : state.metrics.slice(0, 2));
  }

  /* ------------------------------------------------------------------ */
  /* 指标选择                                                            */
  /* ------------------------------------------------------------------ */
  function renderMetrics() {
    const box = $('metrics');
    box.innerHTML = '';
    if (!state.metrics.length) {
      box.innerHTML = '<span class="empty small">没有可画的数值字段</span>';
      return;
    }
    for (const metric of state.metrics) {
      const on = state.selected.has(metric);
      const el = document.createElement('button');
      el.className = 'metric' + (on ? ' on' : '');
      el.innerHTML = `<i style="background:${color_of(metric)}"></i>${escapeHtml(metric)}`;
      el.onclick = () => {
        if (state.selected.has(metric)) state.selected.delete(metric);
        else state.selected.add(metric);
        renderMetrics();
        draw();
      };
      box.appendChild(el);
    }
  }

  /* ------------------------------------------------------------------ */
  /* 曲线                                                                */
  /* ------------------------------------------------------------------ */
  function series() {
    const out = [];
    for (const metric of state.metrics) {
      if (!state.selected.has(metric)) continue;
      const raw = [];
      state.episodes.forEach((row, index) => {
        const value = dig(row, metric);
        if (typeof value === 'number' && Number.isFinite(value)) {
          // 横轴用"第几条记录"而不是 episode 字段: 同一个 out_dir 被多次训练追加写入时,
          // episode 会从 1 重新开始, 拿它当 x 轴会把 165 条压成 2 个点。
          raw.push({ x: index + 1, y: value, ep: row.episode });
        }
      });
      if (!raw.length) continue;
      out.push({ name: metric, color: color_of(metric), points: smooth(raw, state.smooth) });
    }
    return out;
  }

  function smooth(points, window) {
    if (window <= 0) return points;
    const out = [];
    let sum = 0;
    const queue = [];
    for (const point of points) {
      queue.push(point.y);
      sum += point.y;
      if (queue.length > window) sum -= queue.shift();
      out.push({ x: point.x, y: sum / queue.length });
    }
    return out;
  }

  function draw() {
    const canvas = $('chart');
    const empty = $('chart-empty');
    const data = series();
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 340;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (!data.length) {
      empty.style.display = 'flex';
      state.layout = null;
      return;
    }
    empty.style.display = 'none';

    const style = getComputedStyle(document.documentElement);
    const gridColor = style.getPropertyValue('--grid').trim() || '#ddd';
    const muted = style.getPropertyValue('--muted').trim() || '#888';
    const text = style.getPropertyValue('--text').trim() || '#000';

    const pad = { left: 58, right: 14, top: 16, bottom: 26 };
    const plotW = Math.max(10, width - pad.left - pad.right);
    const plotH = Math.max(10, height - pad.top - pad.bottom);

    let xMin = Infinity; let xMax = -Infinity;
    let yMin = Infinity; let yMax = -Infinity;
    for (const line of data) {
      for (const point of line.points) {
        if (point.x < xMin) xMin = point.x;
        if (point.x > xMax) xMax = point.x;
        if (point.y < yMin) yMin = point.y;
        if (point.y > yMax) yMax = point.y;
      }
    }
    if (!Number.isFinite(xMin)) return;
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) { yMax = yMin + 1; yMin = yMin - 1; }

    // 各自归一化: 让量纲差很大的指标 (回报 vs 通关 0/1) 也能同图比较
    const norm = state.norm;
    if (norm) { yMin = 0; yMax = 1; }
    data.forEach((line) => {
      if (!norm) return;
      let lo = Infinity; let hi = -Infinity;
      for (const point of line.points) {
        if (point.y < lo) lo = point.y;
        if (point.y > hi) hi = point.y;
      }
      if (hi === lo) { hi = lo + 1; }
      line._lo = lo;
      line._hi = hi;
      line._scaled = line.points.map((p) => ({ x: p.x, y: (p.y - lo) / (hi - lo) }));
    });

    const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = (y) => pad.top + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    // 网格
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i += 1) {
      const value = yMin + ((yMax - yMin) * i) / yTicks;
      const y = Math.round(sy(value)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(norm ? value.toFixed(2) : fmt(value), pad.left - 8, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = Math.min(8, Math.max(2, Math.round(plotW / 90)));
    for (let i = 0; i <= xTicks; i += 1) {
      const value = xMin + ((xMax - xMin) * i) / xTicks;
      const x = Math.round(sx(value)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.fillText(xLabel(value), x, pad.top + plotH + 6);
    }

    // 折线
    for (const line of data) {
      const points = norm ? line._scaled : line.points;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = sx(point.x);
        const y = sy(point.y);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = line.color;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      ctx.stroke();

      if (points.length <= 2) {
        ctx.fillStyle = line.color;
        for (const point of points) {
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    state.layout = { pad, plotW, plotH, xMin, xMax, yMin, yMax, sx, sy, data, norm, text };

    function xLabel(value) {
      // episode 是整数序号; 跨度大时不要显示成 1.63 这种小数
      return (xMax - xMin) >= 8 ? String(Math.round(value)) : fmt(value);
    }

    // 悬停十字线
    if (state.hoverX != null) {
      const x = sx(state.hoverX);
      ctx.strokeStyle = muted;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;
      for (const line of data) {
        const points = norm ? line._scaled : line.points;
        const nearest = closest(points, state.hoverX);
        if (!nearest) continue;
        ctx.fillStyle = line.color;
        ctx.beginPath();
        ctx.arc(sx(nearest.x), sy(nearest.y), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function closest(points, x) {
    let best = null;
    let bestDelta = Infinity;
    for (const point of points) {
      const delta = Math.abs(point.x - x);
      if (delta < bestDelta) { bestDelta = delta; best = point; }
    }
    return bestDelta < Infinity ? best : null;
  }

  /* ------------------------------------------------------------------ */
  /* 悬停提示                                                            */
  /* ------------------------------------------------------------------ */
  function onMove(event) {
    const layout = state.layout;
    if (!layout) return;
    const rect = $('chart').getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < layout.pad.left || x > layout.pad.left + layout.plotW) {
      hideTip();
      return;
    }
    const value = layout.xMin + ((x - layout.pad.left) / layout.plotW) * (layout.xMax - layout.xMin);
    state.hoverX = value;

    const rows = [];
    for (const line of state.layout.data) {
      const raw = closest(line.points, value);
      if (raw) rows.push({ name: line.name, color: line.color, value: raw.y, x: raw.x, ep: raw.ep });
    }
    if (!rows.length) { hideTip(); return; }

    const tip = $('tip');
    const xText = (state.layout.xMax - state.layout.xMin) >= 8
      ? String(Math.round(rows[0].x)) : fmt(rows[0].x);
    const ep = rows[0].ep;
    const epText = (ep != null && ep !== rows[0].x) ? ` · episode ${ep}` : '';
    tip.innerHTML = `<div class="muted" style="margin-bottom:4px">#${xText}${epText}</div>` +
      rows.map((row) => `<div class="tip-row"><span><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${row.color};margin-right:6px"></i>${escapeHtml(row.name)}</span><b>${fmt(row.value)}</b></div>`).join('');
    tip.classList.add('on');
    const card = $('chart').parentElement.getBoundingClientRect();
    const tipW = 200;
    let left = event.clientX - rect.left + 14;
    if (left + tipW > card.width) left = event.clientX - rect.left - tipW - 14;
    tip.style.left = `${Math.max(6, left)}px`;
    tip.style.top = `${Math.max(6, event.clientY - rect.top - 10)}px`;
    draw();
  }

  function hideTip() {
    state.hoverX = null;
    $('tip').classList.remove('on');
    draw();
  }

  /* ------------------------------------------------------------------ */
  /* 动作分布 / 评估 / 表格                                               */
  /* ------------------------------------------------------------------ */
  function renderActions() {
    const box = $('actions');
    const entries = Object.entries(state.actions || {})
      .filter(([, count]) => Number(count) > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    if (!entries.length) {
      box.innerHTML = '<span class="empty small">这次 run 没有记录动作直方图</span>';
      return;
    }
    const total = entries.reduce((sum, [, count]) => sum + Number(count), 0) || 1;
    box.innerHTML = entries.map(([name, count]) => {
      const share = Number(count) / total;
      const label = name.replace(/\+shoot$/, '') || '原地';
      return `<div class="act-row">
        <span class="act-name" title="${escapeHtml(name)}">${escapeHtml(label)}</span>
        <span class="act-bar"><span style="width:${(share * 100).toFixed(1)}%"></span></span>
        <span class="act-val">${(share * 100).toFixed(0)}%</span>
      </div>`;
    }).join('');
  }

  function renderEval() {
    const box = $('eval');
    if (!state.evalRows.length) {
      box.innerHTML = '<span class="empty small">没有 eval.jsonl（可选）</span>';
      return;
    }
    const rows = state.evalRows.slice(-8).reverse();
    const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
      .filter((key) => !['score', 'step', 'name'].includes(key)).slice(0, 3);
    box.innerHTML = `<table><thead><tr><th>step</th><th>名称</th><th>score</th>${keys.map((k) => `<th>${escapeHtml(k)}</th>`).join('')}</tr></thead><tbody>` +
      rows.map((row) => `<tr><td>${row.step ?? '—'}</td><td>${escapeHtml(row.name ?? '—')}</td>` +
        `<td><b>${fmt(Number(row.score))}</b></td>` +
        keys.map((k) => `<td>${fmt(Number(row[k]))}</td>`).join('') + '</tr>').join('') +
      '</tbody></table>';
  }

  function renderTable() {
    const box = $('table');
    const rows = state.episodes.slice(-12).reverse();
    if (!rows.length) {
      box.innerHTML = '<span class="empty small">—</span>';
      return;
    }
    const columns = ['return', 'frames', 'steps', 'max_stage', 'misses', 'cleared']
      .filter((key) => rows.some((row) => row[key] !== undefined));
    box.innerHTML = '<table><thead><tr><th>episode</th>' +
      columns.map((key) => `<th>${key}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map((row) => `<tr><td>${row.episode ?? '—'}</td>` +
        columns.map((key) => {
          const value = row[key];
          const cls = key === 'cleared' ? (value ? 'ok' : '') : '';
          return `<td class="${cls}">${typeof value === 'number' ? fmt(value) : (value ?? '—')}</td>`;
        }).join('') + '</tr>').join('') +
      '</tbody></table>';
  }

  /* ------------------------------------------------------------------ */
  /* 实时轮询                                                            */
  /* ------------------------------------------------------------------ */
  function stopLive() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function startLive() {
    stopLive();
    if (!state.live || !state.runId) return;
    state.timer = setInterval(async () => {
      if (!state.runId) return;
      try {
        const since = state.episodes.length;
        const data = await api(`/api/run?id=${encodeURIComponent(state.runId)}&since=${since}`);
        let changed = false;
        if (data.episodes && data.episodes.length) {
          state.episodes = state.episodes.concat(data.episodes);
          changed = true;
        }
        if (data.actions && Object.keys(data.actions).length) state.actions = data.actions;
        if (Array.isArray(data.info?.eval)) state.evalRows = data.info.eval;
        if (Array.isArray(data.metrics) && data.metrics.join('|') !== state.metrics.join('|')) {
          state.metrics = data.metrics;
          for (const metric of state.metrics) {
            if (!state.__known?.has(metric)) state.selected.add(metric);
          }
          state.__known = new Set(state.metrics);
          renderMetrics();
        }
        if (changed) {
          renderActions();
          renderEval();
          renderTable();
          draw();
          setStatus(`实时 · ${state.episodes.length} 局`);
        }
      } catch (err) {
        setStatus('实时中断: ' + err.message);
      }
    }, 2000);
  }

  /* ------------------------------------------------------------------ */
  /* 事件绑定                                                            */
  /* ------------------------------------------------------------------ */
  function boot() {
    $('search').addEventListener('input', (event) => {
      state.filter = event.target.value;
      renderRuns();
    });
    $('refresh').onclick = () => { loadRuns(); setStatus('已重新扫描'); };
    $('show-all').addEventListener('change', (event) => {
      state.showAll = event.target.checked;
      renderRuns();
    });

    $('smooth').addEventListener('input', (event) => {
      state.smooth = Number(event.target.value);
      $('smooth-val').textContent = String(state.smooth);
      draw();
    });
    $('norm').addEventListener('change', (event) => {
      state.norm = event.target.checked;
      draw();
    });
    $('live').addEventListener('change', (event) => {
      state.live = event.target.checked;
      if (state.live) startLive(); else StopLiveAndStatus();
    });

    const chart = $('chart');
    chart.addEventListener('mousemove', onMove);
    chart.addEventListener('mouseleave', hideTip);
    window.addEventListener('resize', () => draw());

    $('theme').onclick = () => {
      const root = document.documentElement;
      const order = ['auto', 'dark', 'light'];
      const next = order[(order.indexOf(root.dataset.theme) + 1) % order.length];
      root.dataset.theme = next;
      $('theme').textContent = next === 'auto' ? '主题' : next;
      draw();
    };

    // 有 ?run= 就打开指定 run, 否则自动打开最近一次 —— 打开就有图, 不用先点左边
    const wanted = new URLSearchParams(window.location.search).get('run');
    loadRuns().then(() => {
      if (state.runId) return;
      if (wanted && state.runs.some((run) => run.id === wanted)) openRun(wanted);
      else if (state.runs.length) openRun(state.runs[0].id);
    });
    setInterval(loadRuns, 15000);
  }

  function StopLiveAndStatus() {
    stopLive();
    setStatus('实时已关闭');
  }

  document.addEventListener('DOMContentLoaded', boot);
})();

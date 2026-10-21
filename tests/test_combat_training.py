import unittest
from touhou_rl_mcp.inputs.dialogue import DialogueAssist
from touhou_rl_mcp.rl.reward import RewardModel
from touhou_rl_mcp.types import GameState


def state(flags=3, score=0, hp=100, spell=43, boss=7, bonus=1000000):
    return GameState(stage=1, in_game=True, lives=2, spell_id=spell, score=score,
        boss_hp=hp, boss_max_hp=1000, bullet_count=0, enemies_alive=0,
        raw={"entities": {"bullets": {"status":"ok"}, "enemies": {"status":"ok"},
             "combat": {"status":"ok", "spell_flags":flags, "spell_id":43,
                        "spell_bonus":bonus, "boss_id":boss}}})


class TestCombat(unittest.TestCase):
    def setUp(self):
        self.reward = RewardModel({"spell_capture_bonus":20, "boss_damage_scale":0.01,
                                   "score_scale":1e-5, "shaping_scale":1})

    def parts(self, a, b):
        return self.reward.compute(a,b)[1]

    def test_capture_edge_and_actual_score(self):
        a, b = state(), state(flags=2,score=1000000,spell=44)
        self.assertEqual(self.parts(a,b)["spell_capture"],20)
        self.assertNotIn("spell_capture",self.parts(b,b))
        self.assertEqual(self.parts(a,b)["score"],10)

    def test_timeout_or_bomb_no_capture(self):
        for flags in (0, 8, 32):
            self.assertNotIn("spell_capture",self.parts(state(),state(flags=flags,score=1000000)))

    def test_missing_score_or_corrupt_read_no_capture(self):
        self.assertNotIn("spell_capture",self.parts(state(),state(flags=2,score=999)))
        b=state(flags=2,score=1000000)
        b.raw["entities"]["combat"]["status"]="error"
        self.assertNotIn("spell_capture",self.parts(state(),b))

    def test_damage_does_not_cross_phase_or_boss(self):
        self.assertEqual(self.parts(state(),state(hp=50))["boss_damage"],0.5)
        self.assertNotIn("boss_damage",self.parts(state(),state(hp=0,spell=44)))
        self.assertNotIn("boss_damage",self.parts(state(),state(hp=0,boss=8)))
        timed = state(hp=0)
        timed.raw["entities"]["combat"]["timeout"] = True
        self.assertNotIn("boss_damage",self.parts(state(),timed))

    def test_dialogue_requires_valid_sustained_empty_field(self):
        assist=DialogueAssist()
        s=state()
        self.assertFalse(assist.update(s,8))    # 8 < 阈值 12, 还不算对话
        self.assertTrue(assist.update(s,6))     # 累计 14 >= 12
        s.bullet_count=1
        self.assertFalse(assist.update(s,60))
        s.bullet_count=0
        s.raw["entities"]["bullets"]["status"]="error"
        self.assertFalse(assist.update(s,60))
        s=state()
        s.lives=-1
        self.assertFalse(assist.update(s,60))

    def test_dialogue_has_no_alive_farming(self):
        s=state()
        s.raw["dialogue_assist"]=True
        r=RewardModel({"alive_bonus":1,"shaping_scale":1})
        self.assertNotIn("alive",r.compute(s,s)[1])

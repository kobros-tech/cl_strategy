from skill_memory.cl.skill_memory_plugin import SkillMemoryPlugin


def test_skill_memory_plugin_does_not_own_evaluation_routing():
    plugin = SkillMemoryPlugin(verbose=False)

    assert not hasattr(plugin, "eval_routing")

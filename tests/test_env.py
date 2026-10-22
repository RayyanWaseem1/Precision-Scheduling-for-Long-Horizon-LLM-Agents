from hazprec.agent.toolcall import ToolCall, format_tool_call, parse_tool_call
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.phases import Phase


def run_oracle(env: HorizonGymEnv) -> list:
    results = []
    while not env.state.done:
        results.append(env.step(env.expected))
    return results


def test_oracle_solves_every_task():
    for n_fields in (1, 2, 3):
        env = HorizonGymEnv(HorizonGymConfig(n_hops=6, n_fields=n_fields))
        for seed in range(20):
            env.reset(seed)
            results = run_oracle(env)
            assert results[-1].success
            assert len(results) == env.config.horizon == 13


def test_phases_and_observations():
    env = HorizonGymEnv(HorizonGymConfig(n_hops=3, filler_lines=2))
    task = env.reset(7)
    assert task.oracle_phases[-1] is Phase.STOP
    assert all(p is Phase.ACT for p in task.oracle_phases[:-1])
    first = env.step(env.expected)
    assert "token:" in first.observation and "next:" in first.observation
    assert first.observation.count("note:") == 2
    # The read observation must reveal everything the next update needs.
    rec = task.chain[0]
    assert rec.token in first.observation


def test_generation_is_deterministic():
    cfg = HorizonGymConfig(n_hops=4)
    a, b = HorizonGymEnv(cfg), HorizonGymEnv(cfg)
    assert a.reset(11).oracle == b.reset(11).oracle
    assert a.reset(11).oracle != a.reset(12).oracle


def test_fatal_mode_ends_on_wrong_argument():
    env = HorizonGymEnv(HorizonGymConfig(n_hops=3, error_mode="fatal"))
    task = env.reset(0)
    env.step(env.expected)  # read
    bad = ToolCall("update", {**env.expected.arguments, "ack": task.chain[0].old_token})
    res = env.step(bad)
    assert not res.correct and res.done and not res.success
    assert "ack" in res.reason


def test_recoverable_mode_allows_retries():
    env = HorizonGymEnv(HorizonGymConfig(n_hops=2, error_mode="recoverable", max_retries=2))
    env.reset(3)
    res = env.step(ToolCall("read", {"record_id": "r_0000"}))
    assert not res.correct and not res.done
    res = env.step(env.expected)
    assert res.correct and res.retries_used == 1
    # Exhaust retries on the next step.
    for _ in range(2):
        assert not env.step(None).done
    assert env.step(None).done and not env.state.success


def test_argument_normalization():
    env = HorizonGymEnv(HorizonGymConfig(n_hops=2, n_fields=2))
    env.reset(5)
    env.step(env.expected)
    args = dict(env.expected.arguments)
    args["seq"] = str(args["seq"])  # "1" should equal 1
    args["extra"] = "ignored"
    assert env.step(ToolCall("update", args)).correct


def test_toolcall_roundtrip_and_robustness():
    call = ToolCall("update", {"record_id": "r_1", "ack": "ABCDE", "seq": 2})
    assert parse_tool_call("thinking...\n" + format_tool_call(call)) == call
    assert parse_tool_call('<tool_call>{"name": "read", "arguments": {"record_id": "r_9"}}') == ToolCall(
        "read", {"record_id": "r_9"}
    )
    assert parse_tool_call('{"name": "submit", "arguments": "{\\"count\\": 3}"}') == ToolCall("submit", {"count": 3})
    assert parse_tool_call("no call here") is None
    assert parse_tool_call("<tool_call>{broken json</tool_call>") is None

from reco.cli import main


def test_health_cmd(capsys):
    code = main(["health"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out == "ok"
from backend import wakeword


def test_parse_env_list_splits_and_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    value = " first.ppn , , ~/second.ppn "

    result = wakeword._parse_env_list(value)

    assert result == ["first.ppn", str(tmp_path / "second.ppn")]


def test_porcupine_kwargs_default_keyword():
    kwargs = wakeword._porcupine_create_kwargs(
        0.3,
        keywords_env=None,
        keyword_paths_env=None,
        access_key=None,
    )

    assert kwargs["keywords"] == ["porcupine"]
    assert kwargs["sensitivities"] == [0.3]
    assert "access_key" not in kwargs


def test_porcupine_kwargs_prefers_keyword_paths(tmp_path):
    first = tmp_path / "hej.ppn"
    second = tmp_path / "kompis.ppn"
    kwargs = wakeword._porcupine_create_kwargs(
        1.4,
        keywords_env="ignored",
        keyword_paths_env=f"{first},{second}",
        access_key="abc",
    )

    assert kwargs["keyword_paths"] == [str(first), str(second)]
    assert kwargs["sensitivities"] == [1.0, 1.0]
    assert kwargs["access_key"] == "abc"
    assert "keywords" not in kwargs


def test_ensure_int16_clamps_samples():
    pcm = wakeword._ensure_int16([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    assert list(map(int, pcm)) == [
        -32767,
        -32767,
        -16383,
        0,
        16383,
        32767,
        32767,
    ]

from pitchan.textproc import analyze_text, clean_pron, split_moras


def test_split_moras():
    assert split_moras("ケンキュー") == ["ケ", "ン", "キュ", "ー"]
    assert split_moras("ガッコー") == ["ガ", "ッ", "コ", "ー"]


def test_clean_pron():
    assert clean_pron("シ’") == "シ"
    assert clean_pron("マス’") == "マス"
    assert clean_pron("ヲ") == "オ"


def test_analyze_text_basic():
    aps = analyze_text("私は山梨大学で、音声を研究しています。")
    kanas = [ap.kana for ap in aps]
    assert kanas[0] == "ワタシワ"
    assert kanas[1] == "ヤマナシダイガクデ"
    # 読点の直後の AP は follows_pause
    assert aps[0].follows_pause is True
    assert aps[1].follows_pause is False
    assert aps[2].follows_pause is True
    # モーラ数とアクセント型がラベルから取れている
    assert aps[0].mora_count == 4
    assert all(ap.accent_type is not None for ap in aps)
    # オンセー は 1 型
    onsee = next(ap for ap in aps if ap.kana.startswith("オンセー"))
    assert onsee.accent_type == 1


def test_analyze_text_empty():
    assert analyze_text("") == []
    assert analyze_text("、、。") == []


def test_read_text_file_encodings(tmp_path):
    from pitchan.textproc import analyze_text_file

    for enc in ("utf-8", "utf-8-sig", "cp932"):
        p = tmp_path / f"{enc}.txt"
        p.write_text("私は学生です。", encoding=enc)
        aps = analyze_text_file(str(p))
        assert "".join(ap.kana for ap in aps) == "ワタシワガクセーデス"


def test_bunsetsu_groups():
    from pitchan.textproc import bunsetsu_groups

    aps = analyze_text("私は良い天気の日に研究しています。")
    kanas = [
        "".join(w.pron for w in g) for ap in aps for g in bunsetsu_groups(ap)
    ]
    # 私は / 良い / 天気の日に(非自立名詞は付属)/ 研究しています(サ変+する)
    assert "ワタシワ" in kanas
    assert "テンキノヒニ" in kanas
    assert "ケンキューシテイマス" in kanas
    # 付属語だけの文節は生じない
    assert "ワ" not in kanas and "ニ" not in kanas

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声アノテーションアプリケーション
テキストと音声ファイルから、分節音レベルのTextGridを生成します。
"""

import os
import tempfile
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from praatio import textgrid
from aligner import AudioTextAligner, create_word_tier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}


def allowed_file(filename):
    """許可されたファイル拡張子かチェック"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_textgrid(phoneme_segments, word_segments, duration, output_path):
    """
    アライメント結果からTextGridファイルを生成

    Args:
        phoneme_segments: PhonemeSegmentオブジェクトのリスト
        word_segments: [(word, start_time, end_time), ...] のリスト
        duration: 音声の総時間（秒）
        output_path: 出力ファイルパス
    """
    # TextGridオブジェクトを作成
    tg = textgrid.Textgrid()

    # 音素レベルのIntervalTierを作成
    phoneme_entries = []
    for seg in phoneme_segments:
        phoneme_entries.append(
            textgrid.Interval(seg.start_time, seg.end_time, seg.phoneme)
        )

    phoneme_tier = textgrid.IntervalTier(
        'phonemes', phoneme_entries, minT=0, maxT=duration
    )
    tg.addTier(phoneme_tier)

    # 単語レベルのIntervalTierを作成
    if word_segments:
        word_entries = []
        for word, start, end in word_segments:
            word_entries.append(textgrid.Interval(start, end, word))

        word_tier = textgrid.IntervalTier(
            'words', word_entries, minT=0, maxT=duration
        )
        tg.addTier(word_tier)

    # TextGridファイルを保存
    tg.save(output_path, format='long_textgrid', includeBlankSpaces=True)

    return output_path


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """音声ファイルとテキストを処理してTextGridを生成"""
    try:
        # ファイルとテキストの取得
        if 'audio' not in request.files:
            return jsonify({'error': '音声ファイルがアップロードされていません'}), 400

        audio_file = request.files['audio']
        text = request.form.get('text', '')

        if audio_file.filename == '':
            return jsonify({'error': '音声ファイルが選択されていません'}), 400

        if not text:
            return jsonify({'error': 'テキストが入力されていません'}), 400

        if not allowed_file(audio_file.filename):
            return jsonify({'error': '許可されていないファイル形式です'}), 400

        # 一時ファイルとして保存
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # WAV形式に変換（必要な場合）
        wav_path = audio_path
        if not filename.lower().endswith('.wav'):
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                   f"{os.path.splitext(filename)[0]}.wav")
            # librosaで読み込んで、wavで保存
            y, sr = librosa.load(audio_path, sr=16000)
            import soundfile as sf
            sf.write(wav_path, y, sr)

        # 音声の長さを取得
        y, sr = librosa.load(wav_path, sr=16000)
        duration = len(y) / sr

        # アライメント実行
        aligner = AudioTextAligner(sample_rate=16000)
        alignment_method = request.form.get('method', 'energy')
        phoneme_segments = aligner.align(wav_path, text, method=alignment_method)

        # 単語レベルのティアを作成
        word_segments = create_word_tier(text, duration)

        # TextGrid生成
        textgrid_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                     f"{os.path.splitext(filename)[0]}.TextGrid")
        create_textgrid(phoneme_segments, word_segments, duration, textgrid_path)

        # ファイルを送信
        return send_file(
            textgrid_path,
            as_attachment=True,
            download_name=f"{os.path.splitext(filename)[0]}.TextGrid",
            mimetype='text/plain'
        )

    except Exception as e:
        return jsonify({'error': f'処理中にエラーが発生しました: {str(e)}'}), 500

    finally:
        # 一時ファイルを削除
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            if 'wav_path' in locals() and wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass


@app.route('/health')
def health():
    """ヘルスチェック"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

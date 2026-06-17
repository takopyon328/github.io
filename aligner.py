#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声テキストアライメントモジュール
より高度なアライメント手法を提供
"""

import numpy as np
import librosa
import pyopenjtalk
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PhonemeSegment:
    """音素セグメント情報"""
    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class AudioTextAligner:
    """音声とテキストのアライメントを行うクラス"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        テキストを音素列に変換

        Args:
            text: 入力テキスト

        Returns:
            音素のリスト
        """
        phonemes = []

        try:
            # pyopenjtalkでフルコンテキストラベルを取得
            labels = pyopenjtalk.extract_fullcontext(text)

            for label in labels:
                # フルコンテキストラベルをパース
                # 形式: xx^xx-phoneme+xx=xx/A:...
                parts = label.split('-')
                if len(parts) > 1:
                    phoneme_part = parts[1].split('+')[0]

                    # ポーズ記号をスキップ
                    if phoneme_part not in ['xx', 'pau', 'sil']:
                        phonemes.append(phoneme_part)

        except Exception as e:
            print(f"音素変換エラー: {e}")
            # フォールバック: 文字レベルで分割
            phonemes = [c for c in text if c.strip()]

        return phonemes

    def calculate_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        音声のエネルギーエンベロープを計算

        Args:
            audio: 音声波形

        Returns:
            エネルギーエンベロープ
        """
        # 短時間フレームでのRMSエネルギーを計算
        frame_length = 512
        hop_length = 256

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        return rms

    def detect_voice_segments(self, audio: np.ndarray,
                             threshold: float = 0.02) -> List[Tuple[float, float]]:
        """
        音声区間を検出

        Args:
            audio: 音声波形
            threshold: エネルギー閾値

        Returns:
            [(start_time, end_time), ...] の音声区間リスト
        """
        # エネルギーエンベロープを計算
        rms = self.calculate_energy_envelope(audio)

        # 閾値以上のフレームを検出
        hop_length = 256
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=hop_length
        )

        # 音声区間を抽出
        is_voice = rms > threshold
        segments = []

        start_idx = None
        for i, (is_v, t) in enumerate(zip(is_voice, times)):
            if is_v and start_idx is None:
                start_idx = i
            elif not is_v and start_idx is not None:
                segments.append((times[start_idx], times[i]))
                start_idx = None

        # 最後のセグメント
        if start_idx is not None:
            segments.append((times[start_idx], times[-1]))

        return segments

    def align_simple(self, audio_path: str, text: str) -> List[PhonemeSegment]:
        """
        シンプルなアライメント手法
        音素を音声区間に均等に割り当て

        Args:
            audio_path: 音声ファイルパス
            text: テキスト

        Returns:
            音素セグメントのリスト
        """
        # 音声を読み込み
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / sr

        # テキストを音素に変換
        phonemes = self.text_to_phonemes(text)

        if not phonemes:
            return []

        # 均等分割
        segments = []
        num_phonemes = len(phonemes)
        time_per_phoneme = duration / num_phonemes

        for i, phoneme in enumerate(phonemes):
            start_time = i * time_per_phoneme
            end_time = (i + 1) * time_per_phoneme

            segments.append(PhonemeSegment(
                phoneme=phoneme,
                start_time=start_time,
                end_time=end_time,
                confidence=0.5  # 低い信頼度
            ))

        return segments

    def align_energy_based(self, audio_path: str, text: str) -> List[PhonemeSegment]:
        """
        エネルギーベースのアライメント
        音声のエネルギー分布を考慮して音素を配置

        Args:
            audio_path: 音声ファイルパス
            text: テキスト

        Returns:
            音素セグメントのリスト
        """
        # 音声を読み込み
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / sr

        # テキストを音素に変換
        phonemes = self.text_to_phonemes(text)

        if not phonemes:
            return []

        # 音声区間を検出
        voice_segments = self.detect_voice_segments(audio)

        if not voice_segments:
            # 音声区間が検出できない場合は均等分割
            return self.align_simple(audio_path, text)

        # 音声区間全体の時間を計算
        total_voice_time = sum(end - start for start, end in voice_segments)

        # 各音素に時間を割り当て
        segments = []
        time_per_phoneme = total_voice_time / len(phonemes)

        current_time = voice_segments[0][0]
        segment_idx = 0

        for phoneme in phonemes:
            start_time = current_time
            end_time = current_time + time_per_phoneme

            # セグメント境界を超えた場合、次のセグメントに移動
            while segment_idx < len(voice_segments) and \
                  end_time > voice_segments[segment_idx][1]:

                # 現在のセグメントの終わりまで
                if segment_idx + 1 < len(voice_segments):
                    segment_idx += 1
                    current_time = voice_segments[segment_idx][0]
                    end_time = current_time + time_per_phoneme
                else:
                    break

            segments.append(PhonemeSegment(
                phoneme=phoneme,
                start_time=start_time,
                end_time=min(end_time, duration),
                confidence=0.7  # 中程度の信頼度
            ))

            current_time = end_time

        return segments

    def align(self, audio_path: str, text: str,
             method: str = 'energy') -> List[PhonemeSegment]:
        """
        音声とテキストのアライメントを実行

        Args:
            audio_path: 音声ファイルパス
            text: テキスト
            method: アライメント手法 ('simple' or 'energy')

        Returns:
            音素セグメントのリスト
        """
        if method == 'simple':
            return self.align_simple(audio_path, text)
        elif method == 'energy':
            return self.align_energy_based(audio_path, text)
        else:
            raise ValueError(f"Unknown alignment method: {method}")


def create_word_tier(text: str, duration: float) -> List[Tuple[str, float, float]]:
    """
    単語レベルのティアを作成

    Args:
        text: テキスト
        duration: 音声の長さ（秒）

    Returns:
        [(word, start_time, end_time), ...] のリスト
    """
    # 形態素解析で単語に分割
    words = []

    try:
        # pyopenjtalkで単語分割
        labels = pyopenjtalk.extract_fullcontext(text)

        # ラベルから単語情報を抽出
        current_word = ""
        for label in labels:
            # 簡易的な処理：音素から単語を推定
            parts = label.split('-')
            if len(parts) > 1:
                phoneme = parts[1].split('+')[0]
                if phoneme not in ['xx', 'pau', 'sil']:
                    current_word += phoneme

        # テキストを空白で分割
        words = text.split()

    except:
        # エラー時は文字単位で分割
        words = list(text)

    if not words:
        words = [text]

    # 均等に時間を割り当て
    word_segments = []
    time_per_word = duration / len(words)

    for i, word in enumerate(words):
        start_time = i * time_per_word
        end_time = (i + 1) * time_per_word
        word_segments.append((word, start_time, end_time))

    return word_segments

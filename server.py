#!/usr/bin/env python3
"""
色カウントサーバー
HTMLからの画像を受け取り、色ごとの塊の数を返す
"""

import json
import base64
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

# 色の設定（HSV範囲）
COLOR_CONFIG = {
    "赤": [
        {"lower": [0, 80, 80], "upper": [10, 255, 255]},
        {"lower": [160, 80, 80], "upper": [180, 255, 255]}  # 赤は色相が0と180付近の両方
    ],
    "青": [
        {"lower": [90, 80, 80], "upper": [130, 255, 255]}
    ],
    "緑": [
        {"lower": [40, 80, 80], "upper": [80, 255, 255]}
    ],
    "黄": [
        {"lower": [20, 80, 80], "upper": [35, 255, 255]}
    ],
    "橙": [
        {"lower": [11, 80, 80], "upper": [19, 255, 255]}
    ],
    "紫": [
        {"lower": [130, 80, 80], "upper": [160, 255, 255]}
    ],
}

# 最小面積（小さなノイズを除外）
MIN_AREA = 200


def count_colors(image_data: bytes) -> dict:
    """画像から色ごとの塊の数を数える"""
    # バイトデータをnumpy配列に変換
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "画像の読み込みに失敗しました"}

    # HSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    results = {}

    for color_name, ranges in COLOR_CONFIG.items():
        # 複数範囲のマスクを合成
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lower = np.array(r["lower"], dtype=np.uint8)
            upper = np.array(r["upper"], dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower, upper)

        # ノイズ除去（モルフォロジー処理）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 連結成分分析（塊を数える）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        # 面積フィルタ（背景=0を除外、小さなノイズを除外）
        count = 0
        blobs = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA:
                count += 1
                blobs.append({
                    "area": int(area),
                    "x": int(stats[i, cv2.CC_STAT_LEFT]),
                    "y": int(stats[i, cv2.CC_STAT_TOP]),
                })

        if count > 0:
            results[color_name] = {"count": count, "blobs": blobs}

    return results


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # ログを抑制

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_POST(self):
        if self.path == "/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body)
                image_b64 = data.get("image", "")
                # base64ヘッダー除去
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                image_bytes = base64.b64decode(image_b64)
                results = count_colors(image_bytes)
                response = json.dumps(results, ensure_ascii=False)
                status = 200
            except Exception as e:
                response = json.dumps({"error": str(e)}, ensure_ascii=False)
                status = 500

            self.send_response(status)
            self._set_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = 8765
    print(f"サーバー起動中... http://localhost:{port}")
    print("終了するには Ctrl+C")
    HTTPServer(("localhost", port), Handler).serve_forever()

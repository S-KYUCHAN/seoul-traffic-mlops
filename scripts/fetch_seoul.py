import argparse
import datetime as dt
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET

def parse_volinfo_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    # 정상 응답 체크
    code = root.findtext(".//RESULT/CODE")
    if code and code != "INFO-000":
        msg = root.findtext(".//RESULT/MESSAGE")
        raise RuntimeError(f"API error: {code} {msg}")

    rows = []
    for r in root.findall(".//row"):
        spot_num = r.findtext("spot_num")
        ymd = r.findtext("ymd")  # YYYYMMDD
        hh = r.findtext("hh")    # HH
        io_type = r.findtext("io_type")
        lane_num = r.findtext("lane_num")
        vol = r.findtext("vol")
        if spot_num and ymd and hh and vol:
            rows.append({
                "spot_num": spot_num,
                "ymd": ymd,
                "hh": int(hh),
                "io_type": io_type,
                "lane_num": lane_num,
                "vol": int(vol),
            })
    return rows

def fetch_one_hour(base_url: str, api_key: str, spot: str, ymd: str, hh: int,
                   start_idx=1, end_idx=1000, retries=3, sleep_sec=0.15) -> int:
    """
    해당 시각(ymd, hh)의 vol 합계 반환
    """
    # openapi.seoul.go.kr:8088/{KEY}/xml/VolInfo/1/5/B-32/20250701/11/
    url = f"{base_url.rstrip('/')}/{api_key}/xml/VolInfo/{start_idx}/{end_idx}/{spot}/{ymd}/{hh}/"
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            rows = parse_volinfo_xml(resp.text)
            # 같은 시각의 row들이 lane/io_type으로 여러개 옴 → 합산
            return sum(r["vol"] for r in rows)
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed fetching {spot} {ymd} {hh}: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--spot", required=True, help="e.g. B-32")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True, help="output csv path")
    ap.add_argument("--base_url", default="http://openapi.seoul.go.kr:8088")
    args = ap.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    # 시간 단위 생성
    start_dt = dt.datetime.combine(start_date, dt.time(0, 0, 0))
    end_dt = dt.datetime.combine(end_date, dt.time(23, 0, 0))  # inclusive
    cur = start_dt

    records = []
    while cur <= end_dt:
        ymd = cur.strftime("%Y%m%d")
        hh = f"{cur.hour:02d}"
        vol_sum = fetch_one_hour(args.base_url, args.api_key, args.spot, ymd, hh)
        records.append({"datetime": cur.strftime("%Y-%m-%d %H:%M:%S"), "vol": vol_sum})
        cur += dt.timedelta(hours=1)
        time.sleep(0.10)  # 호출 간격(너무 빠르면 막힐 수 있어서 살짝)

    df = pd.DataFrame(records)

    # 결측 시간 보정(안전장치)
    df["datetime"] = pd.to_datetime(df["datetime"])
    full_range = pd.date_range(start=start_dt, end=end_dt, freq="h")
    df = df.set_index("datetime").reindex(full_range).fillna(0).rename_axis("datetime").reset_index()
    df["vol"] = df["vol"].astype(int)

    df.to_csv(args.out, index=False)
    print(f"✅ saved: {args.out} (rows={len(df)})")

if __name__ == "__main__":
    main()


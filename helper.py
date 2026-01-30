import requests
import xml.etree.ElementTree as ET
import csv
import math

GPX_URL = "https://geoexport.toolforge.org/gpx?coprimary=all&titles=List_of_rocket_launch_sites"

def load_sites_from_gpx(url: str):
    xml = requests.get(url, timeout=60).text
    root = ET.fromstring(xml)

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    sites = []
    for wpt in root.findall("gpx:wpt", ns):
        lat = float(wpt.attrib["lat"])
        lon = float(wpt.attrib["lon"])
        name_el = wpt.find("gpx:name", ns)
        name = name_el.text.strip() if name_el is not None else "UNKNOWN"
        sites.append({
            "name": name,
            "lat_deg": lat,
            "lon_deg": lon,
            "weight": 1.0,  # 先默认等权；之后你可以替换为年发射频次/吞吐量
        })
    return sites

def calc_k_lat(sites):
    num = den = 0.0
    for s in sites:
        w = float(s.get("weight", 1.0))
        phi = math.radians(float(s["lat_deg"]))
        num += w * math.cos(phi)
        den += w
    return num / den if den > 0 else 0.0

def main():
    sites = load_sites_from_gpx(GPX_URL)
    print("Loaded sites:", len(sites))
    print("k_lat (equal-weight):", calc_k_lat(sites))

    # 输出 CSV，给你的批量仿真直接读取
    with open("global_launch_sites_wiki.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lat_deg", "lon_deg", "weight"])
        writer.writeheader()
        writer.writerows(sites)

if __name__ == "__main__":
    main()

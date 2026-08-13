import io
import zipfile

import pandas as pd

from preprocessing.acquire import normalize_pin, pull_acs, pull_cook_county, pull_cta


def test_normalize_pin():
    assert normalize_pin("123456789") == "00000123456789"
    assert normalize_pin("01-02-003-004-0000") == "01020030040000"
    assert normalize_pin(None) is None
    assert normalize_pin("123456789012345") is None


def test_cook_county_is_paged_and_partitioned(tmp_path):
    calls = []

    def fetch(url, **_):
        calls.append(url)
        if len(calls) == 1:
            return b"pin,year\n123,2023\n456,2023\n"
        return b"pin,year\n"

    result = pull_cook_county("sales", [2023], tmp_path, page_size=2, fetch=fetch)
    assert result.rows == 2
    frame = pd.read_parquet(result.files[0])
    assert frame["pin"].tolist() == ["00000000000123", "00000000000456"]
    assert "year%3D2023" in calls[0]
    assert (tmp_path / "cook_county/sales/year=2023/manifest.json").exists()


def test_acs_builds_tract_geoid(tmp_path):
    header = list({
        "NAME": "name", "B19013_001E": "income", "B25003_002E": "a",
        "B25003_001E": "b", "B25002_003E": "c", "B25002_001E": "d",
        "B01003_001E": "e", "B15003_022E": "f", "B15003_023E": "g",
        "B15003_024E": "h", "B15003_025E": "i", "B15003_001E": "j",
    }) + ["state", "county", "tract"]
    payload = __import__("json").dumps([header, ["x"] * 12 + ["17", "031", "010100"]]).encode()
    result = pull_acs(2023, tmp_path, fetch=lambda *_args, **_kwargs: payload)
    assert pd.read_parquet(result.files[0]).loc[0, "geoid"] == "17031010100"


def test_cta_rejects_unsafe_zip_members(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("stops.txt", "stop_id,stop_name\n1,Test\n")
        archive.writestr("../escape.txt", "no")
    pull_cta(tmp_path, fetch=lambda *_args, **_kwargs: data.getvalue())
    assert (tmp_path / "cta_gtfs/stops.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


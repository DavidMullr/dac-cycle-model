"""Fixed and official weather forcing with interval-end semantics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path

from .contract import MofScreeningError, WEATHER_PATH, file_sha256
from .materials import buck_saturation_pressure_Pa


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as error:
        raise MofScreeningError(
            f"invalid UTC timestamp: {value!r}", field="case.weather.start_utc"
        ) from error


@dataclass(frozen=True)
class WeatherHour:
    interval_start_utc: str
    interval_end_utc: str
    temperature_C: float
    relative_humidity: float
    pressure_Pa: float
    co2_ppm_dry: float = 420.0

    @property
    def water_mole_fraction(self) -> float:
        p_water = (
            self.relative_humidity
            * buck_saturation_pressure_Pa(self.temperature_C)
        )
        if p_water >= self.pressure_Pa:
            raise MofScreeningError(
                "weather water partial pressure exceeds total pressure",
                field="case.weather.relative_humidity",
            )
        return p_water / self.pressure_Pa

    @property
    def station_pressure_Pa(self) -> float:
        return self.pressure_Pa

    @property
    def prepared_buck_water_vapour_partial_Pa(self) -> float:
        return self.relative_humidity * buck_saturation_pressure_Pa(
            self.temperature_C
        )

    @property
    def prepared_buck_water_mole_fraction(self) -> float:
        return self.water_mole_fraction

    def as_dict(self) -> dict:
        return {
            "interval_start_utc": self.interval_start_utc,
            "interval_end_utc": self.interval_end_utc,
            "temperature_C": self.temperature_C,
            "relative_humidity": self.relative_humidity,
            "pressure_Pa": self.pressure_Pa,
            "co2_ppm_dry": self.co2_ppm_dry,
            "water_mole_fraction": self.water_mole_fraction,
            "prepared_buck_water_vapour_partial_Pa": (
                self.prepared_buck_water_vapour_partial_Pa
            ),
            "prepared_law": "Buck 1996",
        }


@dataclass(frozen=True)
class WeatherWindow:
    mode: str
    rows: tuple[WeatherHour, ...]
    source_path: str | None
    source_sha256: str | None
    interval_semantics: str

    def at(self, elapsed_s: float) -> WeatherHour:
        if not self.rows:
            raise MofScreeningError("weather window has no rows")
        index = int(max(elapsed_s, 0.0) // 3600.0)
        return self.rows[min(index, len(self.rows) - 1)]

    def provenance(self) -> dict:
        return {
            "mode": self.mode,
            "rows": len(self.rows),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "interval_semantics": self.interval_semantics,
            "first_interval": self.rows[0].as_dict(),
            "last_interval": self.rows[-1].as_dict(),
        }


def load_weather_window(spec: dict, *, path: Path = WEATHER_PATH) -> WeatherWindow:
    hours = int(spec["hours"])
    if spec["mode"] == "fixed":
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        rows = tuple(
            WeatherHour(
                interval_start_utc=(start + timedelta(hours=index)).isoformat(),
                interval_end_utc=(start + timedelta(hours=index + 1)).isoformat(),
                temperature_C=float(spec["temperature_C"]),
                relative_humidity=float(spec["relative_humidity"]),
                pressure_Pa=float(spec["pressure_Pa"]),
                co2_ppm_dry=float(spec["co2_ppm_dry"]),
            )
            for index in range(hours)
        )
        return WeatherWindow(
            mode="fixed",
            rows=rows,
            source_path=None,
            source_sha256=None,
            interval_semantics="constant hourly intervals",
        )

    start_utc = str(spec["start_utc"])
    start = _parse_utc(start_utc)
    first = start + timedelta(hours=1)
    last = start + timedelta(hours=hours)
    rows: list[WeatherHour] = []
    if not path.is_file():
        raise MofScreeningError(
            "portable official-weather dataset is missing", status=500
        )
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            end = _parse_utc(row["utc_timestamp"])
            if end < first or end > last:
                continue
            if row.get("complete", "").lower() != "true":
                raise MofScreeningError(
                    f"incomplete official weather interval ending {row['utc_timestamp']}",
                    field="case.weather",
                )
            rows.append(
                WeatherHour(
                    interval_start_utc=(end - timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    interval_end_utc=end.isoformat().replace("+00:00", "Z"),
                    temperature_C=float(row["temperature_C"]),
                    relative_humidity=float(row["outdoor_rh_fraction"]),
                    pressure_Pa=float(row["station_pressure_Pa"]),
                    co2_ppm_dry=420.0,
                )
            )
    if len(rows) != hours:
        raise MofScreeningError(
            f"official weather expects {hours} complete intervals after {start_utc}; "
            f"found {len(rows)}",
            field="case.weather.start_utc",
        )
    for previous, current in zip(rows, rows[1:]):
        if _parse_utc(current.interval_end_utc) - _parse_utc(
            previous.interval_end_utc
        ) != timedelta(hours=1):
            raise MofScreeningError(
                "official weather intervals are not contiguous",
                field="case.weather",
            )
    return WeatherWindow(
        mode="official",
        rows=tuple(rows),
        source_path=path.name,
        source_sha256=file_sha256(path),
        interval_semantics="start < utc_timestamp <= end",
    )

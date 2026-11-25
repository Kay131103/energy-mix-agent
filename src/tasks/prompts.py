"""Task prompt builders (shared by data generation, training and evaluation)."""
from __future__ import annotations

import json


def plan_prompt(scen) -> str:
    return (
        "Site energy scenario (12-hour window, INR/kWh):\n"
        f"load_kwh={scen.load_kwh}\n"
        f"grid: tou_rate={scen.grid_tou_rate}, cap_kwh={scen.grid_cap_kwh:.0f}\n"
        f"solar: ppa_price={scen.solar_ppa_price:.2f}, cap_kw={scen.solar_cap_kw:.0f}, "
        f"avail_profile={scen.solar_avail_profile}\n"
        f"wind: ppa_price={scen.wind_ppa_price:.2f}, cap_kw={scen.wind_cap_kw:.0f}, "
        f"avail_profile={scen.wind_avail_profile}\n"
        f"battery: power_kw={scen.battery_power_kw:.0f}, energy_kwh={scen.battery_energy_kwh:.0f}, "
        f"charge_eff={scen.battery_charge_eff:.2f}, discharge_eff={scen.battery_discharge_eff:.2f}, "
        f"initial_soc={scen.battery_initial_soc_kwh:.0f}, opex_per_kwh={scen.battery_opex_per_kwh:.2f}\n"
        f"carbon_free_target_share={scen.carbon_target_share:.2f}\n\n"
        "Output the complete plan JSON now."
    )


def bill_prompt(bill_text: str) -> str:
    return "Extract the bill data.\n\n" + bill_text + "\n\nOutput JSON only."


def plan_json_example(n_hours: int = 12) -> str:
    ex = {"hours": [{"h": h, "grid": 0, "solar": 0, "wind": 0, "chg": 0, "dis": 0, "soc": 0}
                    for h in range(n_hours)]}
    return json.dumps(ex, separators=(",", ":"))

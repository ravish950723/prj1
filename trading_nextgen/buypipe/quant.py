from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from .utils import safe_float, logistic, yn


def compute_quant_features(df: pd.DataFrame, qqq_df: Optional[pd.DataFrame] = None) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {}
    if df is None or df.empty:
        return out

    c = pd.to_numeric(df['close'], errors='coerce')
    r = c.pct_change(fill_method=None)
    v = pd.to_numeric(df.get('volume'), errors='coerce')

    mom_21 = safe_float((c.iloc[-1] / c.iloc[-22] - 1.0) * 100.0 if len(c) >= 22 and c.iloc[-22] else 0.0)
    mom_63 = safe_float((c.iloc[-1] / c.iloc[-64] - 1.0) * 100.0 if len(c) >= 64 and c.iloc[-64] else 0.0)
    realized_vol_21 = safe_float(r.tail(21).std() * np.sqrt(252.0) * 100.0)
    downside_vol_21 = safe_float(r[r < 0].tail(21).std() * np.sqrt(252.0) * 100.0)
    skew_63 = safe_float(r.tail(63).skew())
    kurt_63 = safe_float(r.tail(63).kurt())
    volume_pressure = safe_float((v.tail(5).mean() / v.tail(20).mean()) if len(v.dropna()) >= 20 and safe_float(v.tail(20).mean()) > 0 else 0.0)

    rel_strength = 0.0
    beta_qqq = 0.0
    alpha_20 = 0.0
    if qqq_df is not None and not qqq_df.empty:
        q = pd.to_numeric(qqq_df['close'], errors='coerce')
        qr = q.pct_change(fill_method=None)
        if len(c) >= 21 and len(q) >= 21 and c.iloc[-21] and q.iloc[-21]:
            rel_strength = safe_float(((c.iloc[-1] / c.iloc[-21]) - (q.iloc[-1] / q.iloc[-21])) * 100.0)
        joined = pd.concat([r.rename('sym'), qr.rename('qqq')], axis=1).dropna().tail(63)
        if len(joined) >= 20 and joined['qqq'].var() > 0:
            cov = float(np.cov(joined['sym'], joined['qqq'])[0, 1])
            beta_qqq = safe_float(cov / joined['qqq'].var())
            alpha_20 = safe_float((joined['sym'].mean() - beta_qqq * joined['qqq'].mean()) * 252.0 * 100.0)

    value_proxy = safe_float((c.tail(20).mean() - c.iloc[-1]) / c.tail(20).mean() * 100.0 if len(c) >= 20 and c.tail(20).mean() else 0.0)
    quality_proxy = safe_float((mom_63 - realized_vol_21) / 100.0)
    mean_rev_z = safe_float((c.iloc[-1] - c.tail(20).mean()) / (c.tail(20).std() + 1e-9) if len(c) >= 20 else 0.0)

    composite = logistic(
        0.030 * mom_21 +
        0.020 * mom_63 +
        0.025 * rel_strength +
        0.015 * alpha_20 -
        0.020 * realized_vol_21 -
        0.010 * max(beta_qqq, 0) +
        0.020 * value_proxy +
        0.050 * quality_proxy -
        0.030 * abs(mean_rev_z)
    )

    out.update({
        'QUANT_MOM_21D': round(mom_21, 4),
        'QUANT_MOM_63D': round(mom_63, 4),
        'QUANT_REALIZED_VOL_21D': round(realized_vol_21, 4),
        'QUANT_DOWNSIDE_VOL_21D': round(downside_vol_21, 4),
        'QUANT_SKEW_63D': round(skew_63, 4),
        'QUANT_KURT_63D': round(kurt_63, 4),
        'QUANT_BETA_QQQ': round(beta_qqq, 4),
        'QUANT_ALPHA_20D': round(alpha_20, 4),
        'QUANT_REL_STRENGTH_20D': round(rel_strength, 4),
        'QUANT_VALUE_PROXY': round(value_proxy, 4),
        'QUANT_QUALITY_PROXY': round(quality_proxy, 4),
        'QUANT_MEAN_REV_Z': round(mean_rev_z, 4),
        'QUANT_COMPOSITE_SCORE': round(composite, 6),
        'Volume Pressure': round(volume_pressure, 4),
        'QUANT_LONG_BIAS': yn(composite >= 0.58),
        'QUANT_SHORT_BIAS': yn(composite <= 0.42),
    })
    return out

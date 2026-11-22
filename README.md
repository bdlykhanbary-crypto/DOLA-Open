# DOLA-Open
### Open Research for Time-Series Forecasting
### پژوهش آزاد برای پیش‌بینی سری‌های زمانی

---

# English

## About

DOLA-Open is an experimental machine-learning project exploring time-series forecasting with XGBoost.

Features:
- Time-series feature engineering
- Purged chronological data splits
- XGBoost baseline and quantile models
- Prediction interval calibration
- Walk-forward backtesting
- Comparison with a no-change baseline

**The public demonstration uses synthetic data only. No real market prices or live forecasts are included.**

## Experimental Results

Based on 2,200 simulated daily observations:

| Metric | Result |
|---|---:|
| Features | 19 |
| Training samples | 1,444 |
| Validation samples | 286 |
| Test samples | 317 |
| Test RMSE improvement over no-change | -1.44% |
| Raw interval coverage | 65.93% |
| Calibrated interval coverage | 68.14% |
| Walk-forward predictions | 11 |
| Walk-forward direction accuracy | 45.45% |
| Tuned walk-forward RMSE improvement | 0.20% |

The model has not demonstrated a reliable advantage over the simple baseline. The calibrated interval did not reach its nominal 80% coverage.

## Quick Start

Requirements: Python 3.10+, NumPy, Pandas and XGBoost 3.x.

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python prepare_usd_dataset.py
python purge_splits.py
python convert_data.py
python train_baseline.py
python train_quantiles.py
python evaluate_test.py
python evaluate_quantiles.py
python calibrate_interval.py
python walkforward_backtest.py
```

## Limitations

- Synthetic data does not represent real market behavior.
- The walk-forward sample is small.
- Predictive superiority has not been established.
- Prediction intervals need further calibration.
- This is not financial or investment advice.

Contributions and reproducibility improvements are welcome.

---

# فارسی

## دربارهٔ پروژه

DOLA-Open یک پروژهٔ پژوهشی و آزمایشی در زمینهٔ یادگیری ماشین و پیش‌بینی سری‌های زمانی با XGBoost است.

امکانات:
- مهندسی ویژگی برای سری‌های زمانی
- تقسیم زمانی پاک‌سازی‌شده و کاهش نشت اطلاعات
- آموزش مدل پایه و مدل‌های کوانتایلی
- کالیبراسیون بازه‌های پیش‌بینی
- ارزیابی Walk-forward
- مقایسه با روش سادهٔ بدون تغییر

**نسخهٔ عمومی فقط از داده‌های مصنوعی استفاده می‌کند و شامل قیمت واقعی بازار یا پیش‌بینی زنده نیست.**

## نتایج آزمایشی

نتایج حاصل از ۲۲۰۰ مشاهدهٔ روزانهٔ شبیه‌سازی‌شده:

| معیار | نتیجه |
|---|---:|
| تعداد ویژگی‌ها | ۱۹ |
| نمونه‌های آموزش | ۱۴۴۴ |
| نمونه‌های اعتبارسنجی | ۲۸۶ |
| نمونه‌های آزمون | ۳۱۷ |
| بهبود RMSE آزمون نسبت به روش ساده | منفی ۱٫۴۴٪ |
| پوشش بازهٔ خام | ۶۵٫۹۳٪ |
| پوشش بازهٔ کالیبره‌شده | ۶۸٫۱۴٪ |
| پیش‌بینی‌های Walk-forward | ۱۱ |
| دقت تشخیص جهت | ۴۵٫۴۵٪ |
| بهبود RMSE نسخهٔ تنظیم‌شده | ۰٫۲۰٪ |

مدل هنوز برتری قابل‌اتکایی نسبت به روش ساده نشان نداده است. پوشش بازهٔ کالیبره‌شده نیز به هدف اسمی ۸۰ درصد نرسیده است.

## اجرای سریع

پیش‌نیازها: Python 3.10 یا جدیدتر، NumPy، Pandas و XGBoost نسخهٔ ۳.

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python prepare_usd_dataset.py
python purge_splits.py
python convert_data.py
python train_baseline.py
python train_quantiles.py
python evaluate_test.py
python evaluate_quantiles.py
python calibrate_interval.py
python walkforward_backtest.py
```

## محدودیت‌ها

- دادهٔ مصنوعی نمایندهٔ رفتار واقعی بازار نیست.
- تعداد پیش‌بینی‌های Walk-forward کم است.
- برتری پیش‌بینی مدل هنوز اثبات نشده است.
- بازه‌های پیش‌بینی به کالیبراسیون بیشتر نیاز دارند.
- این پروژه توصیهٔ مالی یا سرمایه‌گذاری نیست.

از مشارکت پژوهشی، گزارش خطاها و پیشنهادهای بهبود استقبال می‌کنیم.

---

**DOLA-Open — Open Research, Transparent Results**

**دولا — پژوهش آزاد، نتایج شفاف**

License: Not yet selected. Public visibility does not itself grant reuse rights.
مجوز استفاده هنوز انتخاب نشده است.

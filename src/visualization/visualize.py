from matplotlib.figure import Figure


def plot_forecast(dates, y_true, y_pred, baseline, output_path, horizon_days):
    # Figure OO (bukan pyplot) supaya aman dijalankan di CI tanpa display.
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(dates, y_true, label="Aktual", color="black")
    ax.plot(dates, y_pred, label="Model")
    ax.plot(dates, baseline, label="Baseline naif", linestyle="--")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel(f"Rata-rata page_views {horizon_days} hari ke depan")
    ax.set_title("Forecast vs Aktual (periode test)")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox


sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 14,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    # "font.size": 24,
}
plt.rcParams.update(params)


def mnist(
    ds, legend_title=None, file_path=None,
):
    plt.rcParams.update(
        {
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
        }
    )
    zoom = 0.72
    samples = 150
    x = ds.phi
    y = ds.y
    t = ds.t
    tau_true = ds.tau
    markers = ds.x.reshape((-1, 28, 28))
    markers = (markers - markers.min()) / (markers.max() - markers.min())
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(1080 / 150, 1080 / 150),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    density_axis = ax[0]
    data_axis = ax[1]
    control_color = "C0"
    treatment_color = "C4"

    idx_0 = np.argsort(x[t == 0].ravel())
    idx_1 = np.argsort(x[t == 1].ravel())

    _ = sns.histplot(
        x=x[t == 0][idx_0].ravel(),
        bins=np.arange(-3.2, 3.22, 0.02),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=density_axis,
    )
    _ = sns.histplot(
        x=x[t == 1][idx_1].ravel(),
        bins=np.arange(-3, 3.02, 0.02),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=density_axis,
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_xlim([-3.2, 3.2])
    _ = density_axis.legend(loc="upper left")

    tau_true = 1.0 * (tau_true / tau_true.max())
    sample = np.random.choice(np.arange(len(idx_0)), replace=False, size=(samples,))
    cmap_0 = ListedColormap(["#FFFFFF00", control_color])
    markers_0 = [
        OffsetImage(image, cmap=cmap_0, zoom=zoom)
        for image in markers[t == 0][idx_0][sample]
    ]
    _ = sns.scatterplot(
        x=x[t == 0][idx_0].ravel()[sample],
        y=y[t == 0][idx_0].ravel()[sample],
        markers=markers_0,
        color=control_color,
        s=0,
        ax=data_axis,
    )

    for image, x_coord, y_coord in zip(
        markers_0, x[t == 0][idx_0].ravel()[sample], y[t == 0][idx_0].ravel()[sample]
    ):
        ab = AnnotationBbox(image, (x_coord, y_coord), xycoords="data", frameon=False)
        data_axis.add_artist(ab)
        data_axis.update_datalim([(x_coord, y_coord)])
        data_axis.autoscale()

    sample = np.random.choice(np.arange(len(idx_1)), replace=False, size=(samples,))
    cmap_1 = ListedColormap(["#FFFFFF00", treatment_color])
    markers_1 = [
        OffsetImage(image, cmap=cmap_1, zoom=zoom)
        for image in markers[t == 1][idx_1][sample]
    ]
    _ = sns.scatterplot(
        x=x[t == 1][idx_1].ravel()[sample],
        y=y[t == 1][idx_1].ravel()[sample],
        markers=markers_1,
        color=treatment_color,
        s=0,
        ax=data_axis,
    )

    for image, x_coord, y_coord in zip(
        markers_1, x[t == 1][idx_1].ravel()[sample], y[t == 1][idx_1].ravel()[sample]
    ):
        ab = AnnotationBbox(image, (x_coord, y_coord), xycoords="data", frameon=False)
        data_axis.add_artist(ab)
        data_axis.update_datalim([(x_coord, y_coord)])
        data_axis.autoscale()
    _ = data_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = data_axis.set_xlabel("Covariate embedding $\phi$")
    _ = data_axis.set_ylabel(r"Outcome")
    _ = data_axis.set_xlim([-3.2, 3.2])

    _ = plt.savefig(file_path, dpi=150)
    _ = plt.close()


def dataset(
    ds, legend_title=None, file_path=None,
):
    plt.rcParams.update(
        {
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
        }
    )
    x = ds.x
    y = ds.y
    t = ds.t
    tau_true = ds.tau
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(1080 / 150, 1080 / 150),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    density_axis = ax[0]
    data_axis = ax[1]
    control_color = "C0"
    treatment_color = "C4"

    idx_0 = np.argsort(x[t == 0].ravel())
    idx_1 = np.argsort(x[t == 1].ravel())

    _ = sns.histplot(
        x=x[t == 0][idx_0].ravel(),
        bins=np.arange(-3.2, 3.22, 0.02),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=density_axis,
    )
    _ = sns.histplot(
        x=x[t == 1][idx_1].ravel(),
        bins=np.arange(-3, 3.02, 0.02),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=density_axis,
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_xlim([-3.2, 3.2])
    _ = density_axis.legend(loc="upper left")

    tau_true = 1.0 * (tau_true / tau_true.max())

    sample = np.random.choice(np.arange(len(idx_0)), replace=False, size=(1000,))
    _ = sns.scatterplot(
        x=x[t == 0][idx_0].ravel()[sample],
        y=y[t == 0][idx_0].ravel()[sample],
        color=control_color,
        label="Control",
        ax=data_axis,
    )

    sample = np.random.choice(np.arange(len(idx_1)), replace=False, size=(1000,))
    _ = sns.scatterplot(
        x=x[t == 1][idx_1].ravel()[sample],
        y=y[t == 1][idx_1].ravel()[sample],
        color=treatment_color,
        label="Treated",
        ax=data_axis,
    )
    _ = data_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = data_axis.set_xlabel("Covariate $\mathrm{x}$")
    _ = data_axis.set_ylabel(r"Outcome")
    _ = data_axis.set_xlim([-3.2, 3.2])

    _ = plt.savefig(file_path, dpi=150)
    _ = plt.close()


def acquisition_hist(
    x_pool,
    t_pool,
    x_acquired,
    t_acquired,
    tau_true,
    tau_pred,
    domain,
    legend_title=None,
    file_path=None,
):
    plt.rcParams.update(
        {
            "text.color": "0.2",
            "font.weight": "bold",
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "axes.labelsize": 24,
            "axes.labelcolor": "0.2",
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
        }
    )
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(1920 / 150, 1080 / 150),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 1, 3]},
    )
    density_axis = ax[0]
    acquire_axis = ax[1]
    data_axis = ax[2]
    control_color = "C0"
    treatment_color = "C4"
    function_color = "#ad8bd6"

    idx = np.argsort(x_pool.ravel())
    idx_0 = np.argsort(x_pool[t_pool == 0].ravel())
    idx_1 = np.argsort(x_pool[t_pool == 1].ravel())

    _ = density_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = density_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = sns.histplot(
        x=x_pool[t_pool == 0][idx_0].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=density_axis,
    )
    _ = sns.histplot(
        x=x_pool[t_pool == 1][idx_1].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=density_axis,
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = density_axis.set_ylabel("count")
    _ = density_axis.set_xlim([-3.0, 3.0])
    _ = density_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title="Pool Data"
    )

    _ = data_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = data_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = data_axis.plot(
        x_pool[idx].ravel(),
        tau_true[idx].ravel(),
        color="black",
        lw=4,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    tau_mean = tau_pred.mean(0)
    tau_2sigma = 2 * tau_pred.std(0)
    _ = data_axis.plot(
        domain,
        tau_mean,
        color=function_color,
        lw=2,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$",
    )
    _ = data_axis.fill_between(
        x=domain,
        y1=tau_mean - tau_2sigma,
        y2=tau_mean + tau_2sigma,
        color=function_color,
        alpha=0.3,
    )
    _ = sns.despine()
    _ = data_axis.set_xlabel("Covariate $\mathbf{x}$")
    _ = data_axis.set_ylabel(r"Treatment Effect $\tau$")
    _ = data_axis.tick_params(axis="x", direction="in", pad=-20)
    _ = data_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = data_axis.set_xlim([-3.0, 3.0])
    _ = data_axis.set_ylim([-8, 12])
    _ = data_axis.legend(loc="upper left", bbox_to_anchor=(1, 1.02))

    _ = acquire_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = acquire_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = sns.histplot(
        x=x_acquired[t_acquired == 0].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=acquire_axis,
    )
    _ = sns.histplot(
        x=x_acquired[t_acquired == 1].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=acquire_axis,
    )
    _ = acquire_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = acquire_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = acquire_axis.set_ylabel("count")
    _ = acquire_axis.set_xlim([-3.0, 3.0])
    _ = acquire_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title=legend_title
    )
    im = plt.imread("assets/oatml.png")
    newax = fig.add_axes([0.84, 0.02, 0.28, 0.28], anchor="SW", zorder=1)
    newax.imshow(im, alpha=0.3)
    newax.axis("off")
    _ = plt.savefig(file_path, dpi=150)
    _ = plt.close()


def acquisition_hist(
    x_pool,
    t_pool,
    x_acquired,
    t_acquired,
    tau_true,
    tau_pred,
    domain,
    legend_title=None,
    file_path=None,
):
    plt.rcParams.update(
        {
            "text.color": "0.2",
            "font.weight": "bold",
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "axes.labelsize": 24,
            "axes.labelcolor": "0.2",
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
        }
    )
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(1920 / 150, 1080 / 150),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 1, 3]},
    )
    density_axis = ax[0]
    acquire_axis = ax[1]
    data_axis = ax[2]
    control_color = "C0"
    treatment_color = "C4"
    function_color = "#ad8bd6"

    idx = np.argsort(x_pool.ravel())
    idx_0 = np.argsort(x_pool[t_pool == 0].ravel())
    idx_1 = np.argsort(x_pool[t_pool == 1].ravel())

    _ = density_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = density_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = sns.histplot(
        x=x_pool[t_pool == 0][idx_0].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=density_axis,
    )
    _ = sns.histplot(
        x=x_pool[t_pool == 1][idx_1].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=density_axis,
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = density_axis.set_ylabel("count")
    _ = density_axis.set_xlim([-3.0, 3.0])
    _ = density_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title="Pool Data"
    )

    _ = data_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = data_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = data_axis.plot(
        x_pool[idx].ravel(),
        tau_true[idx].ravel(),
        color="black",
        lw=4,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    tau_mean = tau_pred.mean(0)
    tau_2sigma = 2 * tau_pred.std(0)
    _ = data_axis.plot(
        domain,
        tau_mean,
        color=function_color,
        lw=2,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$",
    )
    _ = data_axis.fill_between(
        x=domain,
        y1=tau_mean - tau_2sigma,
        y2=tau_mean + tau_2sigma,
        color=function_color,
        alpha=0.3,
    )
    _ = sns.despine()
    _ = data_axis.set_xlabel("Covariate $\mathbf{x}$")
    _ = data_axis.set_ylabel(r"Treatment Effect $\tau$")
    _ = data_axis.tick_params(axis="x", direction="in", pad=-20)
    _ = data_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = data_axis.set_xlim([-3.0, 3.0])
    _ = data_axis.set_ylim([-8, 12])
    _ = data_axis.legend(loc="upper left", bbox_to_anchor=(1, 1.02))

    _ = acquire_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = acquire_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = sns.histplot(
        x=x_acquired[t_acquired == 0].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=acquire_axis,
    )
    _ = sns.histplot(
        x=x_acquired[t_acquired == 1].ravel(),
        bins=np.arange(-6, 6.04, 0.04),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=acquire_axis,
    )
    _ = acquire_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = acquire_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = acquire_axis.set_ylabel("count")
    _ = acquire_axis.set_xlim([-3.0, 3.0])
    _ = acquire_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title=legend_title
    )
    im = plt.imread("assets/oatml.png")
    newax = fig.add_axes([0.84, 0.02, 0.28, 0.28], anchor="SW", zorder=1)
    newax.imshow(im, alpha=0.3)
    newax.axis("off")
    _ = plt.savefig(file_path, dpi=150)
    _ = plt.close()


def acquisition_clean(
    x_pool,
    t_pool,
    x_acquired,
    t_acquired,
    tau_true,
    tau_pred,
    domain,
    legend_title=None,
    file_path=None,
):
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(482 / 72, 512 / 72),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 1, 3]},
    )
    density_axis = ax[0]
    acquire_axis = ax[1]
    data_axis = ax[2]
    control_color = "C0"
    treatment_color = "C4"
    function_color = "C9"

    idx = np.argsort(x_pool.ravel())
    idx_0 = np.argsort(x_pool[t_pool == 0].ravel())
    idx_1 = np.argsort(x_pool[t_pool == 1].ravel())

    _ = density_axis.axvspan(-3.5, 2, facecolor=control_color, alpha=0.05)
    _ = density_axis.axvspan(-2, 3.5, facecolor=treatment_color, alpha=0.05)
    _ = sns.kdeplot(
        x=x_pool[t_pool == 0][idx_0].ravel(),
        color=control_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}_{\mathrm{pool}}}(\mathbf{x} | \mathrm{control})$",
        ax=density_axis,
    )
    _ = sns.kdeplot(
        x=x_pool[t_pool == 1][idx_1].ravel(),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}_{\mathrm{pool}}}(\mathbf{x} | \mathrm{treated})$",
        ax=density_axis,
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_ylabel("pool density")
    _ = density_axis.set_xlim([-3.5, 3.5])
    _ = density_axis.legend(loc="upper left")

    _ = data_axis.axvspan(-3.5, 2, facecolor=control_color, alpha=0.05)
    _ = data_axis.axvspan(-2, 3.5, facecolor=treatment_color, alpha=0.05)
    _ = data_axis.plot(
        x_pool[idx].ravel(),
        tau_true[idx].ravel(),
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    tau_mean = tau_pred.mean(0)
    tau_2sigma = 2 * tau_pred.std(0)
    _ = data_axis.plot(
        domain,
        tau_mean,
        color=function_color,
        lw=4,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$, $\mathbf{\omega} \sim q(\mathbf{\Omega} \mid \mathcal{D})$",
    )
    _ = data_axis.fill_between(
        x=domain,
        y1=tau_mean - tau_2sigma,
        y2=tau_mean + tau_2sigma,
        color=function_color,
        alpha=0.3,
    )
    _ = sns.despine()
    _ = data_axis.set_xlabel("covariate value: $\mathbf{x}$")
    _ = data_axis.set_ylabel("CATE")
    _ = data_axis.tick_params(axis="x", direction="in", pad=-20)
    _ = data_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = data_axis.set_xlim([-3.5, 3.5])
    _ = data_axis.set_ylim([-7, 12])
    _ = data_axis.legend(loc="upper left", title=legend_title)

    _ = acquire_axis.axvspan(-3.5, 2, facecolor=control_color, alpha=0.05)
    _ = acquire_axis.axvspan(-2, 3.5, facecolor=treatment_color, alpha=0.05)
    _ = sns.kdeplot(
        x=x_acquired[t_acquired == 0].ravel(),
        color=control_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}_{\mathrm{train}}}(\mathbf{x} | \mathrm{control})$",
        ax=acquire_axis,
    )
    _ = sns.kdeplot(
        x=x_acquired[t_acquired == 1].ravel(),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}_{\mathrm{train}}}(\mathbf{x} | \mathrm{treated})$",
        ax=acquire_axis,
    )
    _ = acquire_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = acquire_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = acquire_axis.set_ylabel("train density")
    _ = acquire_axis.set_xlim([-3.5, 3.5])
    _ = acquire_axis.legend(loc="upper left")
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def functions(
    x,
    t,
    domain,
    tau_true,
    tau_mean,
    legend_title=None,
    legend_loc=None,
    file_path=None,
):
    legend_loc = (0.07, 0.68) if legend_loc is None else legend_loc
    tau_mean = 1.5 * (tau_mean / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=300)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.4,
        hatch="X",
        stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        domain, tau_true, color="black", lw=6, ls=":", label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.plot(
        domain,
        tau_mean.mean(0),
        color="C0",
        lw=2,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$, $\mathbf{\omega} \sim q(\mathbf{\omega} \mid \mathcal{D})$",
    )
    _ = plt.plot(domain, tau_mean.transpose(1, 0), color="C0", lw=1, ls="-", alpha=0.3,)
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-4.2, 4.2])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    _ = plt.legend(loc=legend_loc, title=legend_title)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def acquisition(
    x, t, tau_true, bald, legend_title=None, legend_loc=None, file_path=None,
):
    legend_loc = (0.07, 0.55) if legend_loc is None else legend_loc
    idx = idx = np.argsort(x.ravel())
    idx_0 = np.argsort(x[t == 0].ravel())
    idx_1 = np.argsort(x[t == 1].ravel())
    bald = 1.5 * (bald / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())[idx]
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=300)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.2,
        hatch="X",
        stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        x[idx],
        tau_true,
        color="black",
        lw=6,
        ls=":",
        alpha=0.5,
        label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.scatter(
        x[t == 0][idx_0],
        bald[t == 0][idx_0],
        color="C0",
        s=128,
        alpha=0.1,
        label=r"$\mu\mathrm{-BALD} \mid \mathrm{t} = 0$",
    )
    _ = plt.scatter(
        x[t == 1][idx_1],
        bald[t == 1][idx_1],
        color="C1",
        s=128,
        alpha=0.1,
        label=r"$\mu\mathrm{-BALD} \mid \mathrm{t} = 1$",
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-4.2, 4.2])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    # _ = plt.legend(loc=legend_loc, title=legend_title)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def errorbar(
    x,
    y,
    y_err,
    x_label,
    y_label,
    marker_label=None,
    x_pad=-20,
    y_pad=-45,
    legend_loc="upper left",
    file_path=None,
):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    plt.errorbar(
        x,
        y,
        yerr=y_err,
        linestyle="None",
        marker="o",
        elinewidth=1.0,
        capsize=2.0,
        label=marker_label,
    )
    lim = max(np.abs(x.min()), np.abs(x.max())) * 1.1
    r = np.arange(-lim, lim, 0.1)
    _ = plt.plot(r, r, label="Ground Truth")
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.ylim([-lim, lim])
    _ = plt.legend(loc=legend_loc)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()

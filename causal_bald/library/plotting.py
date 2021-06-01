import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def dataset(
    x,
    t,
    domain,
    tau_true,
    legend_title=None,
    legend_loc=None,
    file_path=None,
):
    legend_loc = (0.07, 0.55) if legend_loc is None else legend_loc
    tau_true = 1.0 * (tau_true / tau_true.max())
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=300)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.8,
        # stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-0.9, 1.2])
    _ = plt.xlim([-3.8, 3.5])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = plt.legend(loc=legend_loc, title=legend_title)
    # _ = plt.savefig(file_path, dpi=300)
    # _ = plt.close()


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
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
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
    _ = plt.plot(
        domain,
        tau_mean.transpose(1, 0),
        color="C0",
        lw=1,
        ls="-",
        alpha=0.3,
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-4.2, 4.2])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    _ = plt.legend(loc=legend_loc, title=legend_title)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def acquisition(
    x,
    t,
    tau_true,
    bald,
    legend_title=None,
    legend_loc=None,
    file_path=None,
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

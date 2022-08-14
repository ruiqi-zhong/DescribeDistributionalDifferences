import numpy as np
import random
import string

from . import colors


def values_min_max(values, base_values):
    """Used to pick our axis limits."""
    fx = base_values + values.sum()
    xmin = fx - values[values > 0].sum()
    xmax = fx - values[values < 0].sum()
    cmax = max(abs(values.min()), abs(values.max()))
    d = xmax - xmin
    xmin -= 0.1 * d
    xmax += 0.1 * d

    return xmin, xmax, cmax


def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return np.array(values), clustering


def get_highlight(values, base_values, fx, tokens, uuid, xmin, xmax, output_name):
    red = tuple(colors.red_rgb * 255)
    light_red = (255, 195, 213)
    return


def text(
    shap_values,
    num_starting_labels=0,
    grouping_threshold=0.01,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
    display=True,
):
    def process_shap_values(
        tokens,
        values,
        grouping_threshold,
        separator,
        clustering=None,
        return_meta_data=False,
    ):
        M = len(tokens)
        if len(values) != M:
            raise Exception
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)
        if return_meta_data:
            return (
                tokens,
                values,
                group_sizes,
                token_id_to_node_id_mapping,
                collapsed_node_ids,
            )
        else:
            return tokens, values, group_sizes

    print("shap values shape: ", len(shap_values.shape))
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None or isinstance(shap_values.output_names, str)
    ):
        print("OPTION 1======")

        xmin = 0
        xmax = 0
        cmax = 0

        for i, v in enumerate(shap_values):
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, group_sizes = process_shap_values(
                v.data, values, grouping_threshold, separator, clustering
            )

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, v.base_values)
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        out = ""
        for i, v in enumerate(shap_values):
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        return out

    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        print("OPTION 2======")
        # xmin_computed = None
        # xmax_computed = None
        # cmax_computed = None

        # for i in range(shap_values.shape[-1]):
        #     values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
        #     tokens, values, group_sizes = process_shap_values(
        #         shap_values[:, i].data,
        #         values,
        #         grouping_threshold,
        #         separator,
        #         clustering,
        #     )

        #     # if i == 0:
        #     #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
        #     #     continue

        #     xmin_i, xmax_i, cmax_i = values_min_max(
        #         values, shap_values[:, i].base_values
        #     )
        #     if xmin_computed is None or xmin_i < xmin_computed:
        #         xmin_computed = xmin_i
        #     if xmax_computed is None or xmax_i > xmax_computed:
        #         xmax_computed = xmax_i
        #     if cmax_computed is None or cmax_i > cmax_computed:
        #         cmax_computed = cmax_i

        # if xmin is None:
        #     xmin = xmin_computed
        # if xmax is None:
        #     xmax = xmax_computed
        # if cmax is None:
        #     cmax = cmax_computed

        # output_values = shap_values.values.sum(0) + shap_values.base_values
        # output_max = np.max(np.abs(output_values))
        # for i, name in enumerate(shap_values.output_names):
        #     scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
        #     color = colors.red_transparent_blue(scaled_value)
        #     color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

    if len(shap_values.shape) == 3:
        print("OPTION 3======")

        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(
                    shap_values[j, :, i]
                )
                tokens, values, group_sizes = process_shap_values(
                    shap_values[j, :, i].data,
                    values,
                    grouping_threshold,
                    separator,
                    clustering,
                )

                xmin_i, xmax_i, cmax_i = values_min_max(
                    values, shap_values[j, :, i].base_values
                )
                if xmin_computed is None or xmin_i < xmin_computed:
                    xmin_computed = xmin_i
                if xmax_computed is None or xmax_i > xmax_computed:
                    xmax_computed = xmax_i
                if cmax_computed is None or cmax_i > cmax_computed:
                    cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = ""
        for i, v in enumerate(shap_values):
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )

        return out
    print("MAIN======")

    xmin, xmax, cmax = values_min_max(shap_values.values, shap_values.base_values)
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(
        shap_values.data, values, grouping_threshold, separator, clustering
    )
    encoded_tokens = [
        t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens
    ]
    output_name = (
        shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    )

    out = ""

    for i, token in enumerate(tokens):
        scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

        out += f"""<span style='background: rgba{color}'>
            {token.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '')}
            </span>"""

    return out

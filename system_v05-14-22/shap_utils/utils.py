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
        print("1")
    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        print("2")
    if len(shap_values.shape) == 3:
        print("3")
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

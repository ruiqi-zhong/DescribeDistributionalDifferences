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


def process_shap_values(
    tokens,
    values,
    grouping_threshold,
    separator,
    clustering=None,
    return_meta_data=False,
):

    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:

        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                + "tokens if shap_values.clustering is None! When passing hierarchical "
                + "attributions the clustering is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = lower_values[li] + lower_values[ri] + values[M + i]
            max_values[i + M] = max(
                abs(values[M + i]) / len(groups[M + i]), max_values[li], max_values[ri]
            )

        # compute the upper_values
        upper_values = np.zeros(len(values))

        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i - M, 0])
            ri = int(clustering[i - M, 1])
            upper_values[i] = value
            value += values[i]
            #             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            #             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)

        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):

            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:

                # compute the dividend at internal nodes
                li = int(clustering[i - M, 0])
                ri = int(clustering[i - M, 1])
                dv = abs(values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if max(max_values[li], max_values[ri]) < dv * grouping_threshold:
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i

                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i

                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens)
    else:
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
    # print("shap values shape: ", len(shap_values.shape))
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None or isinstance(shap_values.output_names, str)
    ):
        # print("OPTION 1======")

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
        out = {}
        for i, v in enumerate(shap_values):
            out = out | text(
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
        # print("OPTION 3======")

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

        out = {}
        for i, v in enumerate(shap_values):
            out = out | text(
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
    # print("MAIN======")

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

    original_string = ""
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
        original_string += token
        out += f"<span style='background: rgba{color}'>{token.replace('<', '&lt;').replace('>', '&gt;').replace(' ##', '')}</span>"

    out_dict = {original_string: {"span": "span"}}
    return out_dict

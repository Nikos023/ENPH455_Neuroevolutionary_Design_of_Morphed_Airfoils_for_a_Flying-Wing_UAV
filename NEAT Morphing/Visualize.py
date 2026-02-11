import graphviz

def draw_net(config, genome, view=False, filename=None):
    dot = graphviz.Digraph(
        format="png",
        graph_attr={
            "rankdir": "TB",
            "splines": "true",
            "nodesep": "0.2",
            "ranksep": "2.5",
            "pad": "0.6",
        },
        node_attr={
            "shape": "circle",
            "fontsize": "12",
            "height": "0.6",
            "width": "0.6",
            "fixedsize": "true"
        }
    )

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    for k in inputs:
        dot.node(
            str(k),
            label=str(k),
            shape="box",
            style="filled",
            fillcolor="lightgray"
        )

    for k in outputs:
        dot.node(
            str(k),
            label=str(k),
            style="filled",
            fillcolor="lightblue"
        )

    for k in genome.nodes:
        if k not in inputs and k not in outputs:
            dot.node(
                str(k),
                label=str(k),
                style="filled",
                fillcolor="white"
            )

    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        dot.edge(
            str(cg.key[0]),
            str(cg.key[1]),
            color="green" if cg.weight > 0 else "red",
            penwidth=str(0.6 + 1.5 * abs(cg.weight))
        )

    dot.render(filename, view=view)
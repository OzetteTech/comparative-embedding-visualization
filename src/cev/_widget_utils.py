import traitlets

_ERR_MESSAGE = "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"

# patched version which allows for numpy comparison
# https://github.com/jupyter-widgets/traittypes/issues/45
class link_widgets(traitlets.link):
    def _update_target(self, change):
        try:
            super()._update_target(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e

    def _update_source(self, change):
        try:
            super()._update_source(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e


diverging_cmap = [
    "#96ffea",
    "#96fee9",
    "#96fde8",
    "#95fbe7",
    "#95fae6",
    "#95f9e5",
    "#95f8e4",
    "#94f6e2",
    "#94f5e1",
    "#94f4e0",
    "#94f3df",
    "#93f1de",
    "#93f0dd",
    "#93efdc",
    "#92eedb",
    "#92ecda",
    "#92ebd9",
    "#92ead8",
    "#91e9d7",
    "#91e7d6",
    "#91e6d5",
    "#91e5d4",
    "#90e4d3",
    "#90e3d1",
    "#90e1d0",
    "#8fe0cf",
    "#8fdfce",
    "#8fdecd",
    "#8fdccc",
    "#8edbcb",
    "#8edaca",
    "#8ed9c9",
    "#8dd8c8",
    "#8dd6c7",
    "#8dd5c6",
    "#8cd4c5",
    "#8cd3c4",
    "#8cd2c3",
    "#8cd0c2",
    "#8bcfc1",
    "#8bcec0",
    "#8bcdbf",
    "#8accbe",
    "#8acabd",
    "#8ac9bc",
    "#89c8bb",
    "#89c7ba",
    "#89c6b8",
    "#88c4b7",
    "#88c3b6",
    "#88c2b5",
    "#87c1b4",
    "#87c0b3",
    "#87beb2",
    "#86bdb1",
    "#86bcb0",
    "#86bbaf",
    "#85baae",
    "#85b8ad",
    "#85b7ac",
    "#84b6ab",
    "#84b5aa",
    "#84b4a9",
    "#83b3a8",
    "#83b1a7",
    "#83b0a6",
    "#82afa5",
    "#82aea4",
    "#82ada3",
    "#81aca2",
    "#81aaa1",
    "#80a9a0",
    "#80a89f",
    "#80a79e",
    "#7fa69d",
    "#7fa59c",
    "#7fa39b",
    "#7ea29a",
    "#7ea199",
    "#7ea098",
    "#7d9f97",
    "#7d9e96",
    "#7c9d95",
    "#7c9b94",
    "#7c9a93",
    "#7b9992",
    "#7b9891",
    "#7a9790",
    "#7a968f",
    "#7a948e",
    "#79938d",
    "#79928c",
    "#78918b",
    "#78908a",
    "#788f89",
    "#778e88",
    "#778d87",
    "#768b86",
    "#768a85",
    "#768985",
    "#758884",
    "#758783",
    "#748682",
    "#748581",
    "#748480",
    "#73827f",
    "#73817e",
    "#72807d",
    "#727f7c",
    "#717e7b",
    "#717d7a",
    "#717c79",
    "#707b78",
    "#707977",
    "#6f7876",
    "#6f7775",
    "#6e7674",
    "#6e7573",
    "#6e7472",
    "#6d7371",
    "#6d7270",
    "#6c7170",
    "#6c706f",
    "#6b6e6e",
    "#6b6d6d",
    "#6a6c6c",
    "#6a6b6b",
    "#696a6a",
    "#6b6969",
    "#6c6968",
    "#6e6968",
    "#6f6967",
    "#706967",
    "#726966",
    "#736966",
    "#756965",
    "#766965",
    "#786964",
    "#796964",
    "#7a6963",
    "#7c6963",
    "#7d6962",
    "#7e6962",
    "#806961",
    "#816961",
    "#826860",
    "#846860",
    "#85685f",
    "#86685f",
    "#88685e",
    "#89685e",
    "#8a685d",
    "#8c685d",
    "#8d685c",
    "#8e685c",
    "#8f685b",
    "#91685b",
    "#92685a",
    "#93685a",
    "#946859",
    "#966859",
    "#976758",
    "#986758",
    "#996757",
    "#9a6757",
    "#9c6756",
    "#9d6756",
    "#9e6755",
    "#9f6755",
    "#a06754",
    "#a26754",
    "#a36653",
    "#a46653",
    "#a56652",
    "#a66652",
    "#a76651",
    "#a96651",
    "#aa6650",
    "#ab6650",
    "#ac654f",
    "#ad654f",
    "#ae654e",
    "#af654e",
    "#b1654d",
    "#b2654d",
    "#b3654c",
    "#b4644c",
    "#b5644b",
    "#b6644b",
    "#b7644a",
    "#b9644a",
    "#ba6449",
    "#bb6449",
    "#bc6348",
    "#bd6348",
    "#be6347",
    "#bf6347",
    "#c06346",
    "#c26246",
    "#c36245",
    "#c46244",
    "#c56244",
    "#c66243",
    "#c76143",
    "#c86142",
    "#c96142",
    "#ca6141",
    "#cb6141",
    "#cd6040",
    "#ce6040",
    "#cf603f",
    "#d0603e",
    "#d15f3e",
    "#d25f3d",
    "#d35f3d",
    "#d45f3c",
    "#d55e3c",
    "#d65e3b",
    "#d75e3b",
    "#d95e3a",
    "#da5d39",
    "#db5d39",
    "#dc5d38",
    "#dd5c38",
    "#de5c37",
    "#df5c36",
    "#e05c36",
    "#e15b35",
    "#e25b35",
    "#e35b34",
    "#e45a33",
    "#e55a33",
    "#e75a32",
    "#e85932",
    "#e95931",
    "#ea5930",
    "#eb5830",
    "#ec582f",
    "#ed582e",
    "#ee572e",
    "#ef572d",
    "#f0562d",
    "#f1562c",
    "#f2562b",
    "#f3552b",
    "#f4552a",
    "#f55429",
    "#f75428",
    "#f85428",
    "#f95327",
    "#fa5326",
    "#fb5226",
    "#fc5225",
    "#fd5124",
    "#fe5123",
    "#ff5023",
]
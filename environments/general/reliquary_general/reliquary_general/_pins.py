"""Pinned by scripts/pack_corpus.py: the sha256 of every block's gzip file
as published, its rows per split and mode, and how many of them open the
run as single-turn rows. Written, not edited."""

FILE_SHA256 = {
    "chat": "9d999ee7fa300bba2268d81c3c530033b8db5a1ba499a9ac0bc9b14a72f54ba7",
    "multiturn": "dffb7c47ba9ac28fb55ddf11466cc761c7c92addad1b829a118b3702b034bce7",
    "ifeval": "99fe7dbff6e2d6771777d9af66ff2c2b41774da4088dd8ff187282d4598a732c",
    "structured": "d3e9f8fff1ab6f020ba6d55098e604cd521b016b09e0df174ecbb4a26da85fa4",
    "safety": "66f03b059ecee281bbba0a7ec1bbbb82e88f7a02c67cb79017ed1bd8c3ba5032",
    "clarification": "1b754c8dea26eefaa938367d05d226bb5bf0bec4e3e9f6aae055ee9e86c2be90",
    "identity": "d4d632c666068bb1b5830e74df436c4a099bb9ac2f251db64416156302daafe3",
    "tools_call": "ae4567f7067a5c82e364b0dfad1d9ab993d0442190a657e329ec447cd452371a",
    "tools_pivot": "c8d03117f330e0408d76e6cbf8fb858ae5564e00acc4943454c4c83a9cba05ae"
}

BLOCK_ROWS = {
    "chat": {
        "train": {
            "direct": 30141,
            "thinking": 21824
        },
        "eval": {
            "direct": 645,
            "thinking": 485
        },
        "qualification": {
            "direct": 607,
            "thinking": 431
        }
    },
    "multiturn": {
        "train": {
            "direct": 8381,
            "thinking": 5945
        },
        "eval": {
            "direct": 176,
            "thinking": 129
        },
        "qualification": {
            "direct": 176,
            "thinking": 116
        }
    },
    "ifeval": {
        "train": {
            "direct": 5429,
            "thinking": 12867
        },
        "eval": {
            "direct": 135,
            "thinking": 262
        },
        "qualification": {
            "direct": 124,
            "thinking": 257
        }
    },
    "structured": {
        "train": {
            "direct": 2953,
            "thinking": 6936
        },
        "eval": {
            "direct": 63,
            "thinking": 138
        },
        "qualification": {
            "direct": 69,
            "thinking": 150
        }
    },
    "safety": {
        "train": {
            "direct": 7302,
            "thinking": 5387
        },
        "eval": {
            "direct": 145,
            "thinking": 115
        },
        "qualification": {
            "direct": 163,
            "thinking": 108
        }
    },
    "clarification": {
        "train": {
            "direct": 660,
            "thinking": 404
        },
        "eval": {
            "direct": 16,
            "thinking": 10
        },
        "qualification": {
            "direct": 16,
            "thinking": 10
        }
    },
    "identity": {
        "train": {
            "direct": 1413,
            "thinking": 1001
        },
        "eval": {
            "direct": 25,
            "thinking": 14
        },
        "qualification": {
            "direct": 24,
            "thinking": 14
        }
    },
    "tools_call": {
        "train": {
            "direct": 2719,
            "thinking": 3988
        },
        "eval": {
            "direct": 57,
            "thinking": 80
        },
        "qualification": {
            "direct": 58,
            "thinking": 98
        }
    },
    "tools_pivot": {
        "train": {
            "direct": 2133,
            "thinking": 3155
        },
        "eval": {
            "direct": 47,
            "thinking": 66
        },
        "qualification": {
            "direct": 38,
            "thinking": 61
        }
    }
}

BLOCK_SINGLE_TURN = {
    "chat": {
        "train": {
            "direct": 30141,
            "thinking": 21824
        },
        "eval": {
            "direct": 645,
            "thinking": 485
        },
        "qualification": {
            "direct": 607,
            "thinking": 431
        }
    },
    "multiturn": {
        "train": {
            "direct": 0,
            "thinking": 0
        },
        "eval": {
            "direct": 0,
            "thinking": 0
        },
        "qualification": {
            "direct": 0,
            "thinking": 0
        }
    },
    "ifeval": {
        "train": {
            "direct": 4680,
            "thinking": 11088
        },
        "eval": {
            "direct": 122,
            "thinking": 224
        },
        "qualification": {
            "direct": 106,
            "thinking": 222
        }
    },
    "structured": {
        "train": {
            "direct": 2380,
            "thinking": 5527
        },
        "eval": {
            "direct": 45,
            "thinking": 106
        },
        "qualification": {
            "direct": 57,
            "thinking": 118
        }
    },
    "safety": {
        "train": {
            "direct": 7302,
            "thinking": 5387
        },
        "eval": {
            "direct": 145,
            "thinking": 115
        },
        "qualification": {
            "direct": 163,
            "thinking": 108
        }
    },
    "clarification": {
        "train": {
            "direct": 660,
            "thinking": 404
        },
        "eval": {
            "direct": 16,
            "thinking": 10
        },
        "qualification": {
            "direct": 16,
            "thinking": 10
        }
    },
    "identity": {
        "train": {
            "direct": 1413,
            "thinking": 1001
        },
        "eval": {
            "direct": 25,
            "thinking": 14
        },
        "qualification": {
            "direct": 24,
            "thinking": 14
        }
    },
    "tools_call": {
        "train": {
            "direct": 0,
            "thinking": 0
        },
        "eval": {
            "direct": 0,
            "thinking": 0
        },
        "qualification": {
            "direct": 0,
            "thinking": 0
        }
    },
    "tools_pivot": {
        "train": {
            "direct": 0,
            "thinking": 0
        },
        "eval": {
            "direct": 0,
            "thinking": 0
        },
        "qualification": {
            "direct": 0,
            "thinking": 0
        }
    }
}

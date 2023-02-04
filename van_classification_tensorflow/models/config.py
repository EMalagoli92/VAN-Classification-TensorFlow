from typing import Dict


MODELS_CONFIG: Dict[str,dict] = {"van_b0": {"embed_dims": [32, 64, 160, 256],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [3, 3, 5, 2],
                                            "name": "van_b0"
                                            },
                                 "van_b1": {"embed_dims": [64, 128, 320, 512],
                                            "mlp_ratios": [8, 8, 4, 4,],
                                            "depths": [2, 2, 4, 2],
                                            "name": "van_b1"
                                            },
                                 "van_b2": {"embed_dims": [64, 128, 320, 512],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [3, 3, 12, 3],
                                            "name": "van_b2"
                                            },
                                 "van_b3": {"embed_dims": [64, 128, 320, 512],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [3, 5, 27, 3],
                                            "name": "van_b3" 
                                            },
                                 "van_b4": {"embed_dims": [64, 128, 320, 512],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [3, 6, 40, 3],
                                            "name": "van_b4"
                                            },
                                 "van_b5": {"embed_dims": [96, 192, 480, 768],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [3, 3, 24, 3],
                                            "name": "van_b5"
                                            },
                                 "van_b6": {"embed_dims": [96, 192, 384, 768],
                                            "mlp_ratios": [8, 8, 4, 4],
                                            "depths": [6,6,90,6],
                                            "name": "van_b6"
                                            },                                   
                                 }

TF_WEIGHTS_URL: str = ""
API_STATUS = "Initializing..."
RESULTS = {}


API_URL = "https://api.premierecopilot.com/api"

AGENT_TOKEN = None

PENDING_JS_CALLS = {}



# Flag global pour arrêter l'agent
STOP_REQUESTED = False

# ------- Podcast
PODCAST_FREQ_WEIGHTS = {"High": 0.5, "Medium": 0.3, "Low": 0.15, "Very low": 0.05}


FPS_MAPPING = {

    23.976 : [0.0 ,0.04170833333333,0.08341666666667,0.125125,0.16683333333333,0.20854166666667,0.25025,0.29195833333333,0.375375,0.45879166666667,0.54220833333333,0.625625,0.70904166666667,0.79245833333333,0.83416666666667,0.875875,0.91758333333333,0.95929166666667],
    24 : [0.0 ,0.04166666666667,0.08333333333333,0.125,0.16666666666667,0.20833333333333,0.25,0.29166666666667,0.375,0.45833333333333,0.54166666666667,0.625,0.66666666666667,0.75,0.79166666666667,0.83333333333333,0.875,0.91666666666667,0.95833333333333],
    25 : [0.0 ,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.36,0.44,0.52,0.6,0.68,0.76,0.8,0.84,0.88,0.92,0.96],
    29.97 : [0.0, 0.03336666666667,0.06673333333333,0.1001,0.16683333333333,0.2002,0.23356666666667,0.3003,0.36703333333333,0.43376666666667,0.5005,0.6006,0.7007,0.76743333333333,0.8008,0.83416666666667,0.86753333333333,0.9009,0.96763333333333],
    30 : [0.0 ,0.03333333333333,0.1,0.13333333333333,0.16666666666667,0.2,0.23333333333333,0.26666666666667,0.36666666666667,0.43333333333333,0.53333333333333,0.6,0.66666666666667,0.76666666666667,0.8,0.83333333333333,0.9,0.93333333333333,0.96666666666667],
    50 : [0.0 ,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.36,0.44,0.52,0.6,0.68,0.76,0.8,0.84,0.88,0.92,0.96],
    59.94 : [0.0 ,0.03336666666667,0.06673333333333,0.11678333333333,0.16683333333333,0.2002,0.23356666666667,0.28361666666667,0.35035,0.43376666666667,0.51718333333333,0.6006,0.68401666666667,0.75075,0.8008,0.83416666666667,0.86753333333333,0.91758333333333,0.96763333333333],
    60 : [0.0 ,0.03333333333333,0.08333333333333,0.13333333333333,0.16666666666667,0.2,0.23333333333333,0.26666666666667,0.36666666666667,0.43333333333333,0.53333333333333,0.6,0.66666666666667,0.76666666666667,0.8,0.83333333333333,0.88333333333333,0.93333333333333,0.96666666666667],

    
    
}


AGENT_HISTORY = []

# ------- TOOLS FOR EDITING PROJECT STRUCTURE
MODEL_AGENT_NAME = "gemini-2.5-flash"
MODEL_REACT_PROJECT_STRUCTURE = "gemini-2.5-flash"
MODEL_REACT_PROJECT_STRUCTURE_TOOL = "gemini-2.5-flash-lite"




# Définition des outils disponibles
create_bin_tool = {
    "name": "create_bin",
    "description": """
    Create a new bin in the project at the right place. A bin is also called 'folder', 'dossier' or 'bin'
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the bin"
            },
            "parent_nodeId": {
                "type": "string",
                "description": "The nodeId of the parent bin. Use 'root' for the root bin",
            },
        },
        "required": ["name", "parent_nodeId"],
    },
}

delete_bin_tool = {
    "name": "delete_bin",
    "description": "Delete a bin in the project. works only with bins (folders, dossiers, bins) empty. Never delete a bin that contains items (clips, sequences, etc.), or any other kind of item.",
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "The nodeId of the bin to delete",
            },
        },
        "required": ["nodeId"],
    },
}

create_sequence_tool = {
    "name": "create_sequence",
    "description": """
    Create a new sequence in the project at the right place. 
    
    ### RULES:
    * Usually, the prompt start with width and height of the sequence
    * Always respect the width and height of the sequence if given
    * If width and height are not given, choose a STANDARD RESOLUTION based on the purpose of the prompt.  
    
    ### STANDARD RESOLUTIONS:
    - Tik Tok, Insta Reels, Shorts : width 1080, height 1920
    - Youtube : width 1920, height 1080
    - Film : width 3840, height 2160
    - Film Cinematic : width 3840, height 1634
        """,
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the sequence"
            },
            "parent_nodeId": {
                "type": "string",
                "description": "The nodeId of the parent bin. use 'root' for the root bin",
            },
            "videoFrameWidth": {
                "type": "number",
                "description": "The width of the sequence",
            },
            "videoFrameHeight": {
                "type": "number",
                "description": "The height of the sequence",
            },
            "videoDisplayFormat": {
                "type": "string",
                "description": "The display format of the sequence",
                "enum": ["23.976fps", "24fps", "25fps", "30fps", "50fps", "60fps"]
            },
        },
        "required": ["name", "parent_nodeId", "videoFrameHeight", "videoFrameWidth", "videoDisplayFormat"],
    },
}

update_sequence_tool = {
    "name": "update_sequence",
    "description": "Update the settings of a sequence in the project",
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "The nodeId of the sequence to update",
            },
            "videoFrameHeight": {
                "type": "number",
                "description": "The height of the sequence",
            },
            "videoFrameWidth": {
                "type": "number",
                "description": "The width of the sequence",
            },
            "videoDisplayFormat": {
                "type": "string",
                "description": "The display format of the sequence",
                "enum": ["23.976fps", "24fps", "25fps", "29.97fps", "30fps", "48fps", "50fps", "59.94fps", "60fps"]
            },
        },
        "required": ["nodeId", "videoFrameHeight", "videoFrameWidth", "videoDisplayFormat"],
    },
}

clone_sequence_tool = {
    "name": "duplicate_sequence",
    "description": "Clone a sequence in the project",
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "The nodeId of the sequence to duplicate",
            },
            "new_name": {
                "type": "string",
                "description": "The name of the new sequence. Use the suffix - Clone",
            },
        },
        "required": ["nodeId", "new_name"],
    },
}

modify_item_tool = {
    "name": "modify_item",
    "description": """Modify an item in the project,
    - if new_name is provided, it will rename the item
    - if new_parent_path is provided, it will move the item to the new path
    - if it's a bin it will move all the items inside, usefull to move many items
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "The nodeId of the item to move",
            },
            "new_name": {
                "type": "string",
                "description": "The new name of the item",
            },
            "new_parent_nodeId": {
                "type": "string",
                "description": "The nodeId of the new parent bin. Use 'root' for the root bin",
            },
        },
        "required": ["nodeId"],
    },
}

move_batch_tool = {
    "name": "move_batch",
    "description": """Move a batch of items to a new bin.
    - use start_item_nodeId and end_item_nodeId to precise the range of items to move in the project struture you received
    - use target_bin_nodeId to precise the target bin
    
    ### WARNING 
    - everything between start_item_nodeId and end_item_nodeId included will be moved to the target bin. So be careful with the range.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "start_item_nodeId": {
                "type": "string",
                "description": "The nodeId of the first item to move",
            },
            "end_item_nodeId": {
                "type": "string",
                "description": "The nodeId of the last item to move",
            },
            "target_bin_nodeId": {
                "type": "string",
                "description": "The nodeId of the target bin",
            },
        },
        "required": ["start_item_nodeId", "end_item_nodeId", "target_bin_nodeId"],
    }, 
}



EDIT_PROJECT_STRUCTURE_TOOL_LIST = [create_bin_tool, delete_bin_tool, 
            create_sequence_tool, update_sequence_tool, clone_sequence_tool, 
            modify_item_tool, move_batch_tool]



# ------- TOOLS FOR EDITING TIMELINE STRUCTURE


move_item_tool = {
    "name": "move_item",
    "description": """
    Move an item at a precise position in the timeline
    - use end to precise the duration of the item
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "ID": {
                "type": "string",
                "description": "The ID in the timeline of the item to move. NOT the media path. NOT the project nodeId",
            },
            "track_number": {
                "type": "number",
                "description": "The track number to move the item to. Starting from 0",
            },
            "start": {
                "type": "number",
                "description": "the Timeline timing to move the item",
            },
            "end": {
                "type": "number",
                "description": "the Timeline timing where the item ends",
            },
            "ripple": {
                "type": "boolean",
                "description": "If True, the item will ripple the existing items",
                "default": False,
            }
        },
        "required": ["ID", "start"],
    },
}

insert_item_tool = {
    "name": "insert_item",
    "description": """
    Insert an item at a precise start and end in the timeline
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "must be an existing nodeId in the PROJECT context. NOT the media path. NOT the timeline ID",
            },
            "track_index": {
                "type": "number",
                "description": "The track index to insert the clip to. starting from 0",
            },
            "start": {
                "type": "number",
                "description": "the Timeline timing to insert the clip",
            },
            "end": {
                "type": "number",
                "description": "the Timeline timing where the clip ends. Be carefull: the duration must not be greater than the original clip duration",
            },
            "inPoint_source": {
                "type": "number",
                "description": "The inPoint in the source item to use. use it only with music, or speech",
            },
            "outPoint_source": {
                "type": "number",
                "description": "The outPoint in the source item to use. use it only with music, or speech",
            },
            "ripple": {
                "type": "boolean",
                "description": "If True, the item will ripple the existing items",
                "default": False,
            },
        },
        "required": ["nodeId", "start"],
    },
}

delete_item_tool = {
    "name": "delete_item",
    "description": "Delete an item in the timeline",
    "parameters": {
        "type": "object",
        "properties": {
            "ID": {
                "type": "string",
                "description": "The ID on the timeline of the item to delete",
            },
            "ripple": {
                "type": "boolean",
                "description": "If True, the item will ripple the existing items. Most of the time, it's False",
                "default": False,
            },
        },
        "required": ["ID"],
    },
}

add_marker_tool = {
    "name": "add_marker",
    "description": """Add a marker to the timeline at a specific time.
    Markers can be used to mark important points, chapters, or sections.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "time": {
                "type": "number",
                "description": "The time in seconds where the marker should be created",
            },
            "comments": {
                "type": "string",
                "description": "The comment/name of the marker",
            },
            "type": {
                "type": "string",
                "description": "The type of marker",
                "enum": ["Comment", "Chapter", "Segmentation", "WebLink"],
                "default": "Comment"
            },
            "color": {
                "type": "integer",
                "description": "Color index (0-7). 0=Green, 1=Red, 2=Purple, 3=Orange, 4=Yellow, 5=White, 6=Blue, 7=Cyan",
                "minimum": 0,
                "maximum": 7,
                "default": 0
            },
            "end": {
                "type": "number",
                "description": "Optional end time in seconds if the marker spans a duration (must be greater than time)",
            },
        },
        "required": ["time", "comments"],
    },
}

edit_effect_tool = {
    "name": "edit_effect",
    "description": """
    Apply an effect to items in the timeline
    - use the prompt to describe the desired effect
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "effects": {
                "type": "array",
                "description": "List of effects to apply",
                "items": {
                    "type": "object",
                    "properties": {
                        "ID": {
                            "type": "string",
                            "description": "The id of the item to apply the effect to",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to edit the effect",
                        },
                    },
                    "required": ["ID", "prompt"],
                }
            }
        },
        "required": ["effects"],
    },      
}

# razor_tool = {
#     "name": "razor",
#     "description": """
#     Razor the timeline at a specific time
    
#     """,
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "start": {
#                 "type": "number",
#                 "description": "The start time to razor the timeline",
#             },
#             "end": {
#                 "type": "number",
#                 "description": "The end time to razor the timeline",
#             },
#         },
#         "required": ["start", "end"],
#     },
# }
    

add_text_tool = {
    "name": "add_text",
    "description": "Add one or more texts to the timeline",
    "parameters": {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "description": "An array, each item describing a text to add to the timeline.",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to add the text to the timeline. Describe the purpose, context, and the creative details such as: color of the text, presence of a box, background, animation, etc.",
                        },
                        "start": {
                            "type": "number",
                            "description": "The start time to add the text to the timeline.",
                        },
                        "duration": {
                            "type": "number",
                            "description": "The duration of the text to add to the timeline.",
                        },
                    },
                    "required": ["prompt", "start", "duration"],
                }
            }
        },
        "required": ["texts"],
    },
}

modify_text_tool = {
    "name": "modify_text",
    "description": "Modify one text on the timeline",
    "parameters": {
        "type": "object",
        "properties": {
            "ID": {
                "type": "string",
                "description": "The ID of the text to modify",  
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to modify the text. Describe the purpose, context, and the creative details such as: color of the text, presence of a box, background, animation, etc.",
            },
        },
        "required": ["ID", "prompt"],
    },
}




EDIT_TIMELINE_STRUCTURE_TOOL_LIST = [move_item_tool, 
                                     insert_item_tool, 
                                     delete_item_tool, 
                                     add_marker_tool, 
                                     add_text_tool,
                                     modify_text_tool,
                                     #  edit_effect_tool
                                     ]










API_STATUS = "Initializing..."
RESULTS = {}


API_URL = "https://api.premierecopilot.com/api"

AGENT_TOKEN = None

PENDING_JS_CALLS = {}



# Flag global pour arrêter l'agent
STOP_REQUESTED = False
REASONING_QUEUE = None

# ------- Podcast
PODCAST_FREQ_WEIGHTS = {"High": 0.5, "Medium": 0.3, "Low": 0.15, "Very low": 0.05}


# FPS_MAPPING = {

#     23.976 : [0.0 ,0.04170833333333,0.08341666666667,0.125125,0.16683333333333,0.20854166666667,0.25025,0.29195833333333,0.375375,0.45879166666667,0.54220833333333,0.625625,0.70904166666667,0.79245833333333,0.83416666666667,0.875875,0.91758333333333,0.95929166666667],
#     24 : [0.0 ,0.04166666666667,0.08333333333333,0.125,0.16666666666667,0.20833333333333,0.25,0.29166666666667,0.375,0.45833333333333,0.54166666666667,0.625,0.66666666666667,0.75,0.79166666666667,0.83333333333333,0.875,0.91666666666667,0.95833333333333],
#     25 : [0.0 ,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.36,0.44,0.52,0.6,0.68,0.76,0.8,0.84,0.88,0.92,0.96],
#     29.97 : [0.0, 0.03336666666667,0.06673333333333,0.1001,0.16683333333333,0.2002,0.23356666666667,0.3003,0.36703333333333,0.43376666666667,0.5005,0.6006,0.7007,0.76743333333333,0.8008,0.83416666666667,0.86753333333333,0.9009,0.96763333333333],
#     30 : [0.0 ,0.03333333333333,0.1,0.13333333333333,0.16666666666667,0.2,0.23333333333333,0.26666666666667,0.36666666666667,0.43333333333333,0.53333333333333,0.6,0.66666666666667,0.76666666666667,0.8,0.83333333333333,0.9,0.93333333333333,0.96666666666667],
#     50 : [0.0 ,0.04,0.08,0.12,0.16,0.2,0.24,0.28,0.36,0.44,0.52,0.6,0.68,0.76,0.8,0.84,0.88,0.92,0.96],
#     59.94 : [0.0 ,0.03336666666667,0.06673333333333,0.11678333333333,0.16683333333333,0.2002,0.23356666666667,0.28361666666667,0.35035,0.43376666666667,0.51718333333333,0.6006,0.68401666666667,0.75075,0.8008,0.83416666666667,0.86753333333333,0.91758333333333,0.96763333333333],
#     60 : [0.0 ,0.03333333333333,0.08333333333333,0.13333333333333,0.16666666666667,0.2,0.23333333333333,0.26666666666667,0.36666666666667,0.43333333333333,0.53333333333333,0.6,0.66666666666667,0.76666666666667,0.8,0.83333333333333,0.88333333333333,0.93333333333333,0.96666666666667], 
# }


AGENT_HISTORY = []

# ------- TOOLS FOR EDITING PROJECT STRUCTURE
MODEL_AGENT_NAME = "gemini-2.5-flash"
MODEL_REACT_PROJECT_STRUCTURE = "gemini-2.5-flash"
MODEL_REACT_PROJECT_STRUCTURE_TOOL = "gemini-2.5-flash-lite"


MODEL_EFFECT = "gemini-2.5-flash"

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
            "sequenceId": {
                "type": "string",
                "description": "The sequenceId of the sequence to update. NOT the nodeId",
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
        "required": ["sequenceId", "videoFrameHeight", "videoFrameWidth", "videoDisplayFormat"],
    },
}

clone_sequence_tool = {
    "name": "duplicate_sequence",
    "description": "Clone a sequence in the project",
    "parameters": {
        "type": "object",
        "properties": {
            "sequenceId": {
                "type": "string",
                "description": "The sequenceId of the sequence to duplicate. NOT the nodeId",
            },
            "new_name": {
                "type": "string",
                "description": "The name of the new sequence. Use the suffix - Clone",
            },
        },
        "required": ["sequenceId", "new_name"],
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
                "description": "The nodeId of the item to move. NOT the sequenceId",
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
                "description": "The track number to move the item to. Starting from 0 index",
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
    Insert an item at a precise start and end in the timeline. Track is handle automatically
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "must be an existing nodeId in the PROJECT context. NOT the media path. NOT the timeline ID",
            },
            # "track_index": {
            #     "type": "number",
            #     "description": "The track index to insert the clip to. starting from 0",
            # },
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


insert_item_batch_tool = {
    "name": "insert_item_batch",
    "description": """
    Insert mutliple items on multiple following timing. 
    As this, no need to precise end for each item that will be covered by the next item.
    There must be the same number of nodeId and start. 
    The nodeId order is the order items are inserted.
    Usefull to insert many itmes on music beats for exemple
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "list_nodeId": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "must be existing nodeIds in the PROJECT context. NOT the media path. NOT the timeline ID. The order of the nodeId is the order of the items inserted",
            },
            "list_start": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "the Timeline timings to insert the clip. The order of the start is the order of the items inserted",
            },
            "last_end": {
                "type": "number",
                "description": "the Timeline timing where the last clip ends. Be carefull: the duration must not be greater than the original clip duration",
            },
        },
        "required": ["list_nodeId", "list_start", "last_end"],
    }
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
    - available effects are: Resizing, Positioning, Cropping, ColorGrading, Opacity
    - don't try to calculate the effect yourself, just describe the desired effect and the tool will calculate it
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
                            "description": "The prompt to edit the effect. Don't calculate position or size yourself. Just describe the desired effect and the user intent.",
                        },
                    },
                    "required": ["ID", "prompt"],
                }
            }
        },
        "required": ["effects"],
    },      
}

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
                                     insert_item_batch_tool,
                                     delete_item_tool, 
                                     add_marker_tool, 
                                     add_text_tool,
                                     modify_text_tool,
                                     edit_effect_tool
                                     ]








# ------- TOOLS FOR EDITING EFFECTS




lumetri_tool = {
    "name":"Lumetri",
    "description": "to perform color correction",
    "parameters": {
        "type": "object",
        "properties": {
            "Temperature": {
                "type": "number",
                "description": "Color temperature adjustment",
                "minimum": -300,
                "maximum": 300,
                "default": 0
            },
            "Tint": {
                "type": "number",
                "description": "Color tint adjustment",
                "minimum": -300,
                "maximum": 300,
                "default": 0
            },
            "Saturation": {
                "type": "number",
                "description": "Color saturation adjustment",
                "minimum": 0,
                "maximum": 300,
                "default": 100
            },
            "Exposure": {
                "type": "number",
                "description": "Exposure adjustment",
                "minimum": -7,
                "maximum": 7,
                "default": 0
            },
            "Contrast": {
                "type": "number",
                "description": "Contrast adjustment",
                "minimum": -150,
                "maximum": 150,
                "default": 0
            },
            "Highlights": {
                "type": "number",
                "description": "Highlights adjustment",
                "minimum": -150,
                "maximum": 150,
                "default": 0
            },
            "Shadows": {
                "type": "number",
                "description": "Shadows adjustment",
                "minimum": -150,
                "maximum": 150,
                "default": 0
            },
            "Whites": {
                "type": "number",
                "description": "Whites adjustment",
                "minimum": -150,
                "maximum": 150,
                "default": 100
            },
            "Blacks": {
                "type": "number",
                "description": "Blacks adjustment",
                "minimum": -150,
                "maximum": 150,
                "default": 0
            },
        },
    },
}

opacity_tool = {
    "name": "Opacity",
    "description": "to perform opacity adjustment",
    "parameters": {
        "type": "object",
        "properties": {
            "Opacity": {
                "type": "number",
                "description": "Opacity level",
                "minimum": 0,
                "maximum": 100,
                "default": 100
            },
        },
    },
}

motion_tool = {
    "name": "Motion",
    "description": "to perform motion, position, scale, rotation, and crop adjustments",
    "parameters": {
        "type": "object",
        "properties": {
            "Position": {
                "type": "array",
                "description": "Position [X, Y] coordinates. Range -0.5, 1.5",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2,
                "default": [0.5, 0.5]
            },
            "Scale": {
                "type": "number",
                "description": "Scale percentage",
                "minimum": 0,
                "maximum": 1000,
                "default": 100
            },
            "Rotation": {
                "type": "number",
                "description": "Rotation angle in degrees",
                "minimum": 0,
                "maximum": 360,
                "default": 0
            },
            "AnchorPoint": {
                "type": "array",
                "description": "Anchor Point [X, Y] coordinates. Range -0.5, 1.5",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2,
                "default": [0.5, 0.5]
            },
            "CropLeft": {
                "type": "number",
                "description": "Crop from left edge",
                "minimum": 0,
                "maximum": 100,
                "default": 0
            },
            "CropTop": {
                "type": "number",
                "description": "Crop from top edge",
                "minimum": 0,
                "maximum": 100,
                "default": 0
            },
            "CropRight": {
                "type": "number",
                "description": "Crop from right edge",
                "minimum": 0,
                "maximum": 100,
                "default": 0
            },
            "CropBottom": {
                "type": "number",
                "description": "Crop from bottom edge",
                "minimum": 0,
                "maximum": 100,
                "default": 0
            },
            "AntiFlickerFilter": {
                "type": "number",
                "description": "Anti-flicker filter",
                "minimum": 0,
                "maximum": 1,
                "default": 0
            },
        },
    },
}

easy_motion_tool = {
    "name": "EasyMotion",
    "description": """to edit easily position and crop
    define the grid and the position in the grid of your element. 

    Letter are for the vertical axis
    Number are for the horizontal axis
    A0 is bottom left


    Exemple 1: 
    - grid in 2x2 (square grid of 2 rows and 2 columns)
    - element in A0 to A1
    -> the element will occupy all the bottom space

    Exemple 2: 
    - grid in 5x1 (so it's a vertical grid of 5 rows with 1 column)
    - element in C0
    -> the element will occupy the middle of the frame, on the whole width, and 1/5 of the height

    Exemple 3: 
    - grid in 1x1 (whole frame)
    - element in A0
    -> the element will occupy the whole frame
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "GridVerticalTiles": {
                "type": "integer",
                "description": "Number of vertical tiles in the grid",
                "minimum": 1,
                "maximum": 10,
                "default": 1
            },
            "GridHorizontalTiles": {
                "type": "integer",
                "description": "Number of horizontal tiles in the grid",
                "minimum": 1,
                "maximum": 10,
                "default": 1
            },
            "ElementStartTile": {
                "type": "string",
                "description": "Starting tile coordinate (e.g., A0, B1, C2)"
            },
            "ElementEndTile": {
                "type": "string",
                "description": "Ending tile coordinate (e.g., A0, B1, C2)"
            }
        },
        "required" : ["GridVerticalTiles", "GridHorizontalTiles", "ElementStartTile", "ElementEndTile"],
    },
}



EFFECT_TOOL_LIST = [lumetri_tool, 
                    opacity_tool, 
                    # motion_tool, 
                    easy_motion_tool]





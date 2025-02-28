CAPTION_PROMPT = [
    "Provide a factual description highlighting important details in this picture.",
]

CLIP_TEMPLATE = {
    "aid": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "whu-rs19": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "siri-whu": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "eurosat": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "millionaid": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "fmow": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "patternnet": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "nwpu": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "SkyScript_cls": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    "rsicb256": [
        lambda c: f"a satellite photo of {c}.",
        lambda c: f"a satellite image of {c}",
    ],
    # fine-grained classification
    "roof_shape": [
        lambda c: f"a satellite photo of building, {c}.",
        lambda c: f"a satellite image of building, {c}",
    ],
    "smoothness": [
        lambda c: f"a satellite photo of road, {c}.",
        lambda c: f"a satellite image of road, {c}",
    ],
    "surface": [
        lambda c: f"a satellite photo of road, {c}.",
        lambda c: f"a satellite image of road, {c}",
    ],
}


PRE_DEFINED_QUESTION = {
    "basic_visual_recognition": [
        "What is the dominant land cover type in this image? (urban/forest/water/agricultural/barren)",
        "How many distinct types of land cover can you identify in this image?",
        "What percentage of the image is covered by water bodies?",
        "Are there any clouds present in the image? If yes, approximately what percentage of the image is cloud-covered?",
        "What season does this image appear to be taken in? What visual cues support your answer?",
    ],
    "spatial_analysis": [
        "What is the approximate scale of this image? (city-scale/regional/continental)",
        "Describe the spatial distribution of urban areas in relation to natural features.",
        "What patterns of human settlement can you observe? (clustered/dispersed/linear)",
        "How does the terrain influence the distribution of vegetation?",
        "Can you identify any transportation networks? How do they relate to urban development?",
    ],
    "environmental_assessment": [
        "Are there any visible signs of environmental degradation?",
        "Can you identify potential areas of soil erosion?",
        "What evidence of water pollution can you observe?",
        "How healthy does the vegetation appear? What indicators are you using?",
        "Are there any visible impacts of climate change or extreme weather events?",
    ],
    "urban_analysis": [
        "What is the predominant urban development pattern?",
        "Can you identify different types of urban land use (residential/commercial/industrial)?",
        "How well-connected is the transportation infrastructure?",
        "Are there clear boundaries between urban and rural areas?",
        "Can you identify any informal settlements or rapid urbanization patterns?",
    ],
    "agricultural_analysis": [
        "What types of agricultural practices are visible in the image?",
        "How does field size and shape vary across the image?",
        "Can you identify any irrigation systems or water management features?",
        "What is the current stage of crop growth in the visible fields?",
        "Are there any visible patterns of crop rotation or fallow land?",
    ],
    "disaster_assessment": [
        "What evidence of natural disasters can you observe?",
        "How has infrastructure been affected by the disaster?",
        "Can you identify areas at risk of future disasters?",
        "What emergency response activities are visible?",
        "How has the landscape changed post-disaster?",
    ],
    "geological_features": [
        "What major geological formations are visible?",
        "Can you identify any fault lines or tectonic features?",
        "What types of erosional patterns are present?",
        "Are there any visible mining or extraction activities?",
        "How does geology influence vegetation patterns?",
    ],
    "infrastructure_analysis": [
        "What types of energy infrastructure can you identify?",
        "How well-developed is the transportation network?",
        "Can you locate major water management facilities?",
        "What patterns of industrial development are visible?",
        "How does infrastructure density vary across the image?",
    ],
    "temporal_understanding": [
        "What time of day was this image captured? What shadows or lighting conditions support your answer?",
        "Which season is represented in this image? What evidence supports this?",
        "What indications of recent urban development can you identify?",
        "What stage of vegetation growth is visible in different areas?",
        "What evidence of water level fluctuations can be observed from shoreline features?",
    ],
    "advanced_reasoning": [
        "Based on the visible patterns, what are the main economic activities in this region?",
        "How do natural features constrain or enable human development?",
        "What ecosystem services are visible in this image?",
        "How sustainable are the visible land use practices?",
        "What future development challenges might this area face based on current patterns?",
    ],
    "quantitative_assessment": [
        "What is the approximate area covered by different land use types?",
        "How dense is the urban development in different parts of the image?",
        "What is the ratio of built-up area to green space?",
        "How fragmented are the natural habitats?",
        "What is the distribution pattern of settlement sizes?",
    ],
    "color_spectral_analysis": [
        "What does the variation in vegetation color indicate about plant health?",
        "Can you identify areas of bare soil based on color signatures?",
        "What do the color patterns in urban areas suggest about building materials?",
        "How do seasonal changes affect the spectral signatures of different features?",
    ],
}

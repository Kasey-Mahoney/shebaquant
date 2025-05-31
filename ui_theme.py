class ColorScheme(Enum):
    """Predefined color schemes for the application"""
    OCEAN = {
        "primary": "#1F3A93",
        "secondary": "#5C6BC0",
        "accent": "#FF7043",
        "background": "#F5F7FA",
        "text": "#333333",  # Dark gray for good contrast on light background
        "success": "#4CAF50",
        "warning": "#FFC107",
        "danger": "#F44336"
    }
    FOREST = {
        "primary": "#2E7D32",
        "secondary": "#66BB6A",
        "accent": "#FFA000",
        "background": "#F1F8E9",
        "text": "#263238",  # Very dark greenish-gray
        "success": "#388E3C",
        "warning": "#F57C00",
        "danger": "#D32F2F"
    }
    SUNSET = {
        "primary": "#D81B60",
        "secondary": "#EC407A",
        "accent": "#FF9800",
        "background": "#FFF3E0",
        "text": "#2D3748",  # Changed from #4A148C to dark gray for better contrast
        "success": "#7CB342",
        "warning": "#FFB300",
        "danger": "#E53935"
    }
    DARK_MODE = {
        "primary": "#6A1B9A",
        "secondary": "#AB47BC",
        "accent": "#FFA000",
        "background": "#121212",
        "text": "#E0E0E0",  # Light gray for good contrast on dark background
        "success": "#00C853",
        "warning": "#FFAB00",
        "danger": "#FF5252"
    }
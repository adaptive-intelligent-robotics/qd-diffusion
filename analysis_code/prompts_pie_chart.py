import matplotlib.pyplot as plt

# Data structure containing categories and levels
data = [
    {"category": "Abstract", "level": "Basic"},
    {"category": "Abstract", "level": "Basic"},
    {"category": "Abstract", "level": "Basic"},
    {"category": "Vehicles", "level": "Simple Detail"},
    {"category": "Vehicles", "level": "Fine-grained Detail"},
    {"category": "Vehicles", "level": "Fine-grained Detail"},
    {"category": "Illustrations", "level": "Imagination"},
    {"category": "Illustrations", "level": "Basic"},
    {"category": "Illustrations", "level": "Style & Format"},
    {"category": "Arts", "level": "Complex"},
    {"category": "Arts", "level": "Complex"},
    {"category": "Arts", "level": "Style & Format"},
    {"category": "World Knowledge", "level": "Basic"},
    {"category": "World Knowledge", "level": "Basic"},
    {"category": "World Knowledge", "level": "Complex"},
    {"category": "People", "level": "Fine-grained Detail"},
    {"category": "People", "level": "Fine-grained Detail"},
    {"category": "People", "level": "Simple Detail"},
    {"category": "Animals", "level": "Writing & Symbols"},
    {"category": "Animals", "level": "Basic"},
    {"category": "Animals", "level": "Style & Format"},
    {"category": "Artifacts", "level": "Fine-grained Detail"},
    {"category": "Artifacts", "level": "Writing & Symbols"},
    {"category": "Artifacts", "level": "Simple Detail"},
    {"category": "Food & Beverage", "level": "Fine-grained Detail"},
    {"category": "Food & Beverage", "level": "Simple Detail"},
    {"category": "Food & Beverage", "level": "Imagination"},
    {"category": "Produce & Plants", "level": "Basic"},
    {"category": "Produce & Plants", "level": "Style & Format"},
    {"category": "Produce & Plants", "level": "Complex"},
    {"category": "Outdoor Scenes", "level": "Simple Detail"},
    {"category": "Outdoor Scenes", "level": "Quantity"},
    {"category": "Outdoor Scenes", "level": "Linguistic Structures"},
    {"category": "Indoor Scenes", "level": "Complex"},
    {"category": "Indoor Scenes", "level": "Fine-grained Detail"},
    {"category": "Indoor Scenes", "level": "Quantity"}
]

def generate_pie_charts():
    # 1. Calculate counts
    category_counts = {}
    level_counts = {}

    for item in data:
        cat = item["category"]
        lvl = item["level"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    # 2. Setup plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color map
    colors = plt.get_cmap('tab20c')(range(20))

    # 3. Create Category Pie Chart
    wedges1, texts1, autotexts1 = ax1.pie(
        category_counts.values(), 
        labels=category_counts.keys(), 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85
    )
    ax1.set_title("Distribution by Category")

    # Make percent labels smaller
    plt.setp(autotexts1, size=8, weight="bold", color="white")

    # 4. Create Challenge Level Pie Chart
    wedges2, texts2, autotexts2 = ax2.pie(
        level_counts.values(), 
        labels=level_counts.keys(), 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85
    )
    ax2.set_title("Distribution by Challenge Level")
    
    # Make percent labels smaller
    plt.setp(autotexts2, size=8, weight="bold", color="white")

    # Layout adjustment
    plt.tight_layout()
    
    # Save the figure
    output_filename = "figures/prompts_distribution.png"
    plt.savefig(output_filename)
    print(f"Chart saved as {output_filename}")

    # Show plot (optional, depends on environment)
    # plt.show()

if __name__ == "__main__":
    generate_pie_charts()
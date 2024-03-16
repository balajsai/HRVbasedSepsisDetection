# Simpulate data as provided by the user
data = [
    ["Features", "t statistic for Sepsis", "p value for Sepsis"],
    ["Mean.rate", "10.8753397", "3.40E-27"],
    ["Coefficient.of.variation", "-11.176121", "1.31E-28"],
    ["Poincar..SD1", "-15.068272", "4.82E-50"],
    ["Poincar..SD2", "-14.135262", "2.22E-44"],
    # Add the rest of your data here in a similar format
]

# Convert data to markdown table format
markdown_table = "| " + " | ".join(data[0]) + " |\n"
markdown_table += "|---" * len(data[0]) + "|\n"
for row in data[1:]:
    markdown_table += "| " + " | ".join(row) + " |\n"

# The markdown_table variable now contains the Markdown table
markdown_table


import json
from fpdf import FPDF

def export_to_json(strategy, filename="strategy.json"):
    with open(filename, "w") as f:
        json.dump(strategy, f)

def export_to_pdf(lap_data, filename="strategy_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for lap in lap_data:
        pdf.cell(200, 10, txt=f"Lap {lap['Lap']}: {lap['Time']:.2f}s on {lap['Tire']}", ln=True)
    pdf.output(filename)

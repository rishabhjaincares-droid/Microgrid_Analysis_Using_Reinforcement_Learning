from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Smart_Microgrid_Technical_Report.pdf"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"&bull; {text}", style)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#465A69"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#24516F"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11,
            textColor=colors.HexColor("#384955"),
        )
    )
    return styles


def table(data, col_widths=None):
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6F8FA")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C2CC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64717B"))
    page = canvas.getPageNumber()
    canvas.drawRightString(A4[0] - 0.65 * inch, 0.45 * inch, f"Page {page}")
    canvas.drawString(0.65 * inch, 0.45 * inch, "Smart Microgrid Energy Management Technical Report")
    canvas.restoreState()


def build_story():
    styles = make_styles()
    H1 = styles["ReportHeading1"]
    H2 = styles["ReportHeading2"]
    B = styles["Body"]
    BL = styles["ReportBullet"]
    S = styles["Small"]

    story = [
        Spacer(1, 1.0 * inch),
        p("Smart Microgrid Energy Management Using Reinforcement Learning", styles["CoverTitle"]),
        p("Comprehensive Technical Project Report", styles["CoverSubtitle"]),
        p("Based on the microgrid simulation codebase, NASA POWER data pipeline, RL controllers, and evaluation outputs.", styles["CoverSubtitle"]),
        Spacer(1, 0.45 * inch),
        table(
            [
                ["Project Scope", "Grid-connected residential microgrid with PV, battery storage, load, tariffs, and RL control"],
                ["Primary Algorithms", "Deep Q-Network (DQN), Proximal Policy Optimization (PPO), tariff-aware rule-based baseline"],
                ["Evaluation Data", "NASA POWER 2024 hourly solar/temperature data for Hosur, Tamil Nadu"],
                ["Key Outcome", "RL reduced November 2024 bill cost by 12.81% and peak-hour grid import by 54.73%"],
            ],
            [1.55 * inch, 4.65 * inch],
        ),
        PageBreak(),
    ]

    story += [
        p("Abstract", H1),
        p(
            "This project develops and evaluates a smart microgrid energy management system for a grid-connected "
            "residential-scale microgrid with photovoltaic generation, battery energy storage, household load demand, "
            "and time-of-day electricity tariffs. The central objective is to reduce electricity cost and peak-hour grid "
            "dependency by learning optimal battery charge, discharge, and idle decisions under variable solar generation "
            "and demand.",
            B,
        ),
        p(
            "The system is implemented as a custom Gymnasium-compatible simulation environment in microgrid_env.py. "
            "Real hourly solar and temperature data from NASA POWER for Hosur, Tamil Nadu are processed in data_loader.py, "
            "while synthetic residential load profiles are generated using appliance-level demand assumptions in "
            "load_simulator.py. Reinforcement learning controllers based on DQN and PPO are trained using Stable-Baselines3 "
            "and compared against a tariff-aware rule-based baseline.",
            B,
        ),
        p(
            "Simulation results show that the RL controller consistently reduces electricity cost and peak-hour grid import. "
            "For the full November 2024 held-out test period, the RL controller reduced bill cost from Rs 745.11 to Rs 649.68, "
            "a 12.81% saving, and reduced peak-hour grid import by 54.73%.",
            B,
        ),
        p("1. Introduction", H1),
        p(
            "Electrical distribution systems are increasingly incorporating distributed renewable energy resources such as "
            "rooftop solar photovoltaic systems, battery storage, and controllable household loads. While these technologies "
            "can reduce dependence on the utility grid, their effectiveness depends strongly on intelligent energy management. "
            "Solar generation is intermittent, residential demand varies throughout the day, and electricity tariffs may change "
            "according to time-of-day pricing.",
            B,
        ),
        p(
            "A microgrid controller must make sequential decisions: whether to charge the battery, discharge it, or leave it idle "
            "at each time step. A poor control strategy may export useful solar energy at a low tariff, charge the battery at the "
            "wrong time, or discharge too early before high-price evening hours. This project addresses that control problem using "
            "reinforcement learning, where the controller learns a policy from simulation experience rather than relying only on "
            "fixed heuristic rules.",
            B,
        ),
        p("2. Problem Definition", H1),
        p("Problem Statement", H2),
        p(
            "The engineering problem is to design a microgrid energy management controller that minimizes electricity cost and "
            "peak-hour grid import while respecting physical battery constraints and maintaining energy balance between solar "
            "generation, household load, battery storage, and the utility grid.",
            B,
        ),
        bullet("Solar generation is variable and depends on irradiance conditions.", BL),
        bullet("Residential load has morning and evening demand peaks.", BL),
        bullet("Grid import cost varies by time-of-day tariff.", BL),
        bullet("Battery operation is constrained by SOC limits, efficiency losses, charge/discharge limits, and degradation cost.", BL),
        bullet("Exported solar energy receives a lower credit than the import tariff, making local utilization more valuable.", BL),
        p("Relevance", H2),
        p(
            "This problem is relevant to residential and small institutional microgrids in regions with high solar potential and "
            "emerging time-of-day tariff structures. The project uses Hosur, Tamil Nadu solar data, making the simulation "
            "regionally meaningful for Indian microgrid studies.",
            B,
        ),
        p("3. Objectives", H1),
        p("Primary Objectives", H2),
        bullet("Develop a custom microgrid simulation environment with solar PV, battery storage, residential load, grid exchange, and tariff-based billing.", BL),
        bullet("Train reinforcement learning controllers using DQN and PPO for battery energy management.", BL),
        bullet("Compare RL-based control with a rule-based baseline under identical test conditions.", BL),
        bullet("Evaluate performance using bill cost, grid import, exported energy, battery cycling, and peak-hour grid import.", BL),
        p("Secondary Objectives", H2),
        bullet("Use real NASA POWER solar irradiance and temperature data for simulation realism.", BL),
        bullet("Generate deterministic residential load profiles with appliance-style demand components.", BL),
        bullet("Provide a dashboard-based demonstration interface for scenario comparison.", BL),
        bullet("Explore forecasting support through an LSTM solar irradiance model.", BL),
    ]

    story += [
        p("4. System Architecture and Methodology", H1),
        p("Overall Architecture", H2),
        p(
            "The project follows a simulation-based microgrid control architecture: NASA POWER solar and temperature data are "
            "combined with a synthetic residential load model to form an hourly microgrid dataset. This dataset is passed to "
            "SmartMicrogridEnv, where either an RL controller or a tariff-aware rule-based controller selects battery actions. "
            "The resulting logs are summarized into operational and economic metrics.",
            B,
        ),
        p("Microgrid Components", H2),
        bullet("Solar PV generation: 4.0 kW PV capacity, 0.78 performance ratio, and 3.5 kWh inverter limit.", BL),
        bullet("Battery storage: 5.0 kWh capacity, 20% minimum SOC, 95% maximum SOC, 50% starting SOC, 2.0 kWh hourly charge/discharge limit, and 95% efficiency.", BL),
        bullet("Residential load: base demand, refrigeration, fans, lighting, electronics, cooking, pump events, washing events, and small stochastic variation.", BL),
        bullet("Utility grid: imports and exports with Rs 5.60/kWh solar-hour tariff, Rs 7.00/kWh normal tariff, Rs 8.40/kWh peak tariff, and Rs 3.00/kWh export credit.", BL),
        p("State, Action, and Reward Design", H2),
        bullet("State vector: SOC, normalized solar generation, normalized load, normalized tariff, sine hour, cosine hour, and peak-hour indicator.", BL),
        bullet("Action space: high discharge, low discharge, idle, low charge, and high charge.", BL),
        bullet("Reward: negative bill cost minus battery throughput degradation penalty.", BL),
        p("5. Implementation Details", H1),
        p("Simulation Environment", H2),
        p(
            "The core simulation is implemented in the SmartMicrogridEnv class. At each hourly step, the environment supplies "
            "load directly from solar generation, applies the selected battery action, enforces SOC and charge/discharge limits, "
            "calculates grid import and export, computes electricity bill cost, applies battery degradation cost, and stores a "
            "detailed operational log.",
            B,
        ),
        p("Data Pipeline", H2),
        p(
            "The dataset is built from NASA POWER hourly data for 2024. The loader detects the NASA CSV header automatically, "
            "cleans missing values marked as -999, converts irradiance into PV output, adds temporal and tariff fields, joins "
            "the generated load profile, and splits the data into training months 1-9, validation month 10, and test months 11-12.",
            B,
        ),
        p("Control Strategies", H2),
        bullet("Rule-based controller: stores PV surplus, preserves reserve before evening peak hours, charges during useful cheap periods, and discharges mainly during expensive peak periods.", BL),
        bullet("RL controller: loads trained DQN or PPO models from Stable-Baselines3 and predicts deterministic actions from the current observation.", BL),
        bullet("Training: DQN uses an MLP policy, learning rate 1e-4, replay buffer size 50,000, batch size 32, exploration fraction 0.15, and target update interval 5,000. PPO uses an MLP policy, learning rate 1e-4, rollout length 2,048, and batch size 64.", BL),
    ]

    story += [
        p("6. Results and Key Findings", H1),
        p("Reward-Based Controller Comparison", H2),
        table(
            [
                ["Controller", "Mean Reward", "Finding"],
                ["Rule-Based", "-2177.10", "Reference baseline"],
                ["Greedy", "-2960.12", "Worse than baseline due to short-term decisions"],
                ["DQN", "-1000.98", "54.02% reward improvement over rule-based"],
                ["PPO", "-999.47", "54.09% reward improvement over rule-based"],
            ],
            [1.2 * inch, 1.15 * inch, 3.75 * inch],
        ),
        Spacer(1, 8),
        p("Held-Out November 2024 Performance", H2),
        table(
            [
                ["Metric", "RL", "Rule-Based", "Improvement"],
                ["Bill Cost", "Rs 649.68", "Rs 745.11", "12.81% lower"],
                ["Grid Import", "149.34 kWh", "168.21 kWh", "11.22% lower"],
                ["Peak-Hour Grid Import", "10.76 kWh", "23.77 kWh", "54.73% lower"],
                ["Battery Cycles", "22.65", "17.70", "+4.95 cycles"],
            ],
            [1.75 * inch, 1.2 * inch, 1.35 * inch, 1.55 * inch],
        ),
        Spacer(1, 8),
        p(
            "The November evaluation shows that RL achieved lower cost mainly by shifting battery discharge toward high-tariff "
            "evening periods. The RL controller used more battery cycles than the baseline, but this additional utilization "
            "substantially reduced peak-hour grid import. Exported energy was also lower under RL, indicating better local use "
            "of solar energy rather than unnecessary export at a lower credit.",
            B,
        ),
        p("Peak-Week Scenario Performance", H2),
        table(
            [
                ["Scenario", "Bill Cost Reduction", "Grid Import Reduction", "Peak Import Reduction"],
                ["Mid-November peak week", "Rs 27.67 (13.52%)", "13.76%", "56.71%"],
                ["Mid-December peak week", "Rs 27.51 (13.77%)", "12.50%", "61.23%"],
            ],
            [1.9 * inch, 1.45 * inch, 1.45 * inch, 1.45 * inch],
        ),
        Spacer(1, 8),
        p(
            "These results show that the learned controller is especially effective during evening peak demand periods, where "
            "battery scheduling has the highest economic and grid-support value.",
            B,
        ),
    ]

    img = ROOT / "controller_comparison.png"
    if img.exists():
        story += [
            p("Controller Reward Comparison Chart", H2),
            KeepTogether(
                [
                    Image(str(img), width=5.8 * inch, height=3.8 * inch),
                    p("Figure: Reward comparison and relative improvement across baseline, greedy, DQN, and PPO controllers.", S),
                ]
            ),
        ]

    story += [
        p("7. Conclusion and Future Scope", H1),
        p("Conclusion", H2),
        p(
            "The project successfully demonstrates a reinforcement-learning-based energy management system for a residential "
            "smart microgrid. The implemented simulator models the interaction between solar PV generation, battery storage, "
            "household demand, electricity tariffs, and grid exchange. The RL controller achieved better cost and peak-demand "
            "performance than the tariff-aware rule-based baseline on held-out test scenarios.",
            B,
        ),
        p(
            "The most important outcome is that RL did not simply minimize instantaneous grid import; instead, it learned a "
            "temporal control strategy that preserves or charges the battery before expensive hours and uses stored energy during "
            "peak periods. This validates reinforcement learning as a suitable approach for sequential microgrid scheduling problems.",
            B,
        ),
        p("Future Scope", H2),
        bullet("Integrate the LSTM solar forecaster directly into the RL observation space.", BL),
        bullet("Add real household smart-meter data instead of synthetic load profiles.", BL),
        bullet("Extend the model to include controllable loads such as EV charging, water pumping, or HVAC scheduling.", BL),
        bullet("Perform hyperparameter tuning and longer training for DQN/PPO comparison.", BL),
        bullet("Add battery aging models with nonlinear degradation cost.", BL),
        bullet("Deploy the trained controller on edge hardware such as Raspberry Pi or ESP32 for hardware-in-the-loop validation.", BL),
        bullet("Compare RL with model predictive control for a stronger control-theory benchmark.", BL),
        p("8. Outcome of the Work", H1),
        p(
            "The outcome of this work is a complete software prototype for smart microgrid energy management. It includes a "
            "Gymnasium-compatible microgrid simulation environment, a real-data solar pipeline using NASA POWER data, a "
            "deterministic household load simulator, DQN and PPO reinforcement learning training scripts, evaluation scripts for "
            "held-out monthly and scenario-based testing, and a Streamlit dashboard for demonstration and interpretation.",
            B,
        ),
        p("9. Key Source Files Referenced", H1),
        bullet("config.py: battery, PV, tariff, and dataset split configuration.", BL),
        bullet("data_loader.py: NASA POWER ingestion, irradiance conversion, and train/validation/test split.", BL),
        bullet("load_simulator.py: appliance-style residential load model.", BL),
        bullet("microgrid_env.py: custom Gymnasium environment, action space, energy balance, and reward calculation.", BL),
        bullet("controllers.py: rule-based and Stable-Baselines3 controller wrappers.", BL),
        bullet("train.py: DQN and PPO training pipeline.", BL),
        bullet("evaluate.py: controller evaluation and metric summarization.", BL),
        bullet("dashboard.py: Streamlit visualization and scenario demonstration interface.", BL),
    ]
    return story


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Smart Microgrid Energy Management Technical Report",
        author="Codex",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()

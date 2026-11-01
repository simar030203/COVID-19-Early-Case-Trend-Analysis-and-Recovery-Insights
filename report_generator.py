from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch

from datetime import datetime
import os


# ======================================================
# Generate Prediction PDF Report
# ======================================================

def generate_prediction_report(
    country,
    confirmed_year,
    confirmed_month,
    confirmed_day,
    released_flag,
    deceased_flag,
    prediction,
    confidence
):

    filename = "COVID19_Prediction_Report.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]

    normal = styles["BodyText"]

    story = []

    # -------------------------------------------------

    story.append(
        Paragraph(
            "COVID-19 AI Prediction Report",
            title_style
        )
    )

    story.append(Spacer(1,0.3*inch))

    story.append(
        Paragraph(
            f"Generated on : {datetime.now()}",
            normal
        )
    )

    story.append(Spacer(1,0.3*inch))

    # -------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_data = [

        ["Country",country],

        ["Confirmed Year",confirmed_year],

        ["Confirmed Month",confirmed_month],

        ["Confirmed Day",confirmed_day],

        ["Released Flag",released_flag],

        ["Deceased Flag",deceased_flag]

    ]

    table = Table(patient_data,colWidths=[200,250])

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.lightblue),

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),

            ("FONTNAME",(0,0),(-1,-1),"Helvetica"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ])

    )

    story.append(table)

    story.append(Spacer(1,0.4*inch))

    # -------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Result",
            heading_style
        )
    )

    prediction_table = [

        ["Predicted State",prediction],

        ["Confidence",f"{confidence:.2f}%"]

    ]

    table2 = Table(prediction_table,colWidths=[200,250])

    table2.setStyle(

        TableStyle([

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,0),(0,-1),colors.lightgrey),

            ("BACKGROUND",(1,0),(1,-1),colors.beige),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ])

    )

    story.append(table2)

    story.append(Spacer(1,0.4*inch))

    # -------------------------------------------------

    story.append(
        Paragraph(
            "AI Healthcare Recommendation",
            heading_style
        )
    )

    if prediction.lower()=="released":

        recommendation = """

        Patient has high probability of recovery.

        Continue medications.

        Follow regular health monitoring.

        """

    elif prediction.lower()=="isolated":

        recommendation = """

        Patient should remain isolated.

        Monitor symptoms carefully.

        Avoid unnecessary contact.

        """

    else:

        recommendation = """

        Immediate medical intervention recommended.

        Continuous monitoring required.

        Hospital admission advised.

        """

    story.append(
        Paragraph(
            recommendation,
            normal
        )
    )

    story.append(Spacer(1,0.3*inch))

    # -------------------------------------------------

    story.append(
        Paragraph(
            "Project Information",
            heading_style
        )
    )

    story.append(

        Paragraph(

"""
This report was generated automatically by the

COVID-19 AI Analytics System

Developed using

• Machine Learning

• Streamlit

• Random Forest

• Artificial Intelligence

• Data Analytics

""",

normal

)

)

    doc.build(story)

    return filename
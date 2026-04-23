"""Generate a realistic sample support-ticket Excel file for the analyzer demo."""
import random
from datetime import date, timedelta
import pandas as pd

random.seed(42)

OUTPUT_PATH = "/Users/himanshurawat/ticket-analyzer/sample_tickets.xlsx"
N_ROWS = 60

CATEGORY_WEIGHTS = {
    "Password Reset": 25, "Login Issue": 15, "VPN Access": 10,
    "Software Install": 10, "Hardware Failure": 8, "Email Issue": 10,
    "Printer Issue": 8, "Network Slow": 7, "Access Request": 4,
    "Billing Query": 2, "Other": 1,
}
DEPARTMENT_WEIGHTS = {
    "Finance": 30, "Sales": 15, "Operations": 12, "Engineering": 10,
    "Customer Support": 10, "HR": 8, "IT": 8, "Marketing": 7,
}
PRIORITY_WEIGHTS = {"Low": 15, "Medium": 40, "High": 35, "Critical": 10}
STATUS_WEIGHTS = {"Open": 20, "In Progress": 20, "Resolved": 30, "Closed": 30}

DESCRIPTIONS = {
    "Password Reset": [
        "Unable to reset password after SSO migration, getting 500 error on reset link.",
        "Password reset email never arrives in inbox; checked spam folder.",
        "Reset link expires before I can use it, keeps saying invalid token.",
        "Forgot password, need temporary credentials to log in today.",
        "Reset flow loops back to login page without updating password.",
    ],
    "Login Issue": [
        "Outlook keeps prompting for credentials even after resetting password.",
        "Getting 'account locked' message after one failed attempt.",
        "MFA prompt not appearing on mobile, cannot complete login.",
        "SSO redirect fails and returns to corporate portal repeatedly.",
    ],
    "VPN Access": [
        "VPN client disconnects every 5 minutes from home network.",
        "Cannot connect to VPN from new laptop, getting authentication error.",
        "VPN is extremely slow when accessing shared drives.",
    ],
    "Software Install": [
        "Need Adobe Acrobat Pro installed on my workstation for contracts.",
        "Slack installer fails with permission denied error.",
        "Requesting install of Power BI Desktop for quarterly reports.",
    ],
    "Hardware Failure": [
        "Laptop screen flickers and eventually goes black after an hour of use.",
        "Keyboard keys stuck, several letters not registering.",
        "Docking station no longer detects external monitors.",
    ],
    "Email Issue": [
        "Outlook crashes on startup after last Windows update.",
        "Not receiving external emails since yesterday morning.",
        "Calendar invites from clients go straight to junk folder.",
    ],
    "Printer Issue": [
        "Floor 3 printer showing paper jam but there's no paper stuck.",
        "Cannot add network printer, driver install fails silently.",
        "Print jobs queue up but never actually print.",
    ],
    "Network Slow": [
        "Internet in conference room B is unusably slow during calls.",
        "File transfers to shared drive take 10x longer than last week.",
    ],
    "Access Request": [
        "Need access to the FY26 budget SharePoint folder for planning.",
        "Requesting admin rights to install development tools.",
    ],
    "Billing Query": ["Software license invoice shows charges for decommissioned seats."],
    "Other": ["Monitor stand broken, requesting replacement from facilities."],
}


def weighted_choice(weights: dict) -> str:
    return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def main() -> None:
    today = date.today()
    rows = []
    for i in range(1, N_ROWS + 1):
        category = weighted_choice(CATEGORY_WEIGHTS)
        rows.append({
            "Ticket ID": f"TKT-{i:05d}",
            "Category": category,
            "Description": random.choice(DESCRIPTIONS[category]),
            "Department": weighted_choice(DEPARTMENT_WEIGHTS),
            "Date": pd.Timestamp(today - timedelta(days=random.randint(0, 44))),
            "Priority": weighted_choice(PRIORITY_WEIGHTS),
            "Status": weighted_choice(STATUS_WEIGHTS),
        })
    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# agent_tools.py

import logging

# configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def user_points_details(loyalty_membership_number: str) -> str:
    logger.info("Tool get_user_points called with input: %s", loyalty_membership_number)
    loyalty_details = {
        "U1001": 13876,
        "U1002": 1345,
        "U1003": 47890,
        "U1004": 0,
        "U1005": 125000,
    }
    key = loyalty_membership_number.upper()
    if key in loyalty_details:
        points = loyalty_details[key]
        output = f"{key} has {points} loyalty points in their account."
    else:
        output = "User does not exist in the system."
    logger.info("Tool get_user_points output: %s", output)
    return output


def user_ticket_status(ticket_number: str) -> str:
    logger.info("Tool get_ticket_status called with input: %s", ticket_number)
    ticket_details = {
        "KVPMH5": "On Hold",
        "A4BHY5": "Confirmed",
        "KKC2FG": "Cancelled",
        "HHGYUF": "Confirmed",
        "FH674G": "On Hold",
    }
    key = ticket_number.upper()
    if key in ticket_details:
        status = ticket_details[key]
        output = f"{key} is currently in {status} status."
    else:
        output = "Ticket does not exist in the system."
    logger.info("Tool get_ticket_status output: %s", output)
    return output

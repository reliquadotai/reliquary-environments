# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce data models used within AutomationBench."""

from reliquary_automationbench._upstream.schema.salesforce.base import (
    SalesforceRecord,
    SalesforceState,
    generate_salesforce_id,
)
from reliquary_automationbench._upstream.schema.salesforce.account import Account
from reliquary_automationbench._upstream.schema.salesforce.attachment import Attachment
from reliquary_automationbench._upstream.schema.salesforce.contact import Contact
from reliquary_automationbench._upstream.schema.salesforce.email import Email
from reliquary_automationbench._upstream.schema.salesforce.lead import Lead
from reliquary_automationbench._upstream.schema.salesforce.opportunity import Opportunity
from reliquary_automationbench._upstream.schema.salesforce.campaign import Campaign
from reliquary_automationbench._upstream.schema.salesforce.campaign_member import CampaignMember
from reliquary_automationbench._upstream.schema.salesforce.case import Case
from reliquary_automationbench._upstream.schema.salesforce.case_comment import CaseComment
from reliquary_automationbench._upstream.schema.salesforce.document import Document
from reliquary_automationbench._upstream.schema.salesforce.folder import Folder
from reliquary_automationbench._upstream.schema.salesforce.task import Task
from reliquary_automationbench._upstream.schema.salesforce.event import Event
from reliquary_automationbench._upstream.schema.salesforce.note import Note
from reliquary_automationbench._upstream.schema.salesforce.user import User

# Trigger model rebuild so all forward references are resolved
SalesforceState.model_rebuild()

__all__ = [
    "SalesforceRecord",
    "SalesforceState",
    "generate_salesforce_id",
    "Account",
    "Attachment",
    "Contact",
    "Email",
    "Lead",
    "Opportunity",
    "Campaign",
    "CampaignMember",
    "Case",
    "CaseComment",
    "Document",
    "Folder",
    "Task",
    "Event",
    "Note",
    "User",
]

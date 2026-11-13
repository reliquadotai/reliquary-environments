# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn schema model definitions used by AutomationBench."""

from reliquary_automationbench._upstream.schema.linkedin.base import (
    LinkedInRecord,
    LinkedInState,
    generate_linkedin_company_id,
    generate_linkedin_post_id,
    generate_linkedin_urn,
)
from reliquary_automationbench._upstream.schema.linkedin.company import Company
from reliquary_automationbench._upstream.schema.linkedin.connection import Connection
from reliquary_automationbench._upstream.schema.linkedin.group import Group
from reliquary_automationbench._upstream.schema.linkedin.invitation import Invitation
from reliquary_automationbench._upstream.schema.linkedin.job import Job
from reliquary_automationbench._upstream.schema.linkedin.message import Message
from reliquary_automationbench._upstream.schema.linkedin.post import Post, PostContent
from reliquary_automationbench._upstream.schema.linkedin.profile import Education, Experience, Profile

# Resolve forward references declared in LinkedInState after all imports are complete
LinkedInState.model_rebuild()

__all__ = [
    # Base
    "LinkedInRecord",
    "LinkedInState",
    "generate_linkedin_urn",
    "generate_linkedin_post_id",
    "generate_linkedin_company_id",
    # Models
    "Profile",
    "Experience",
    "Education",
    "Connection",
    "Company",
    "Post",
    "PostContent",
    "Job",
    "Group",
    "Invitation",
    "Message",
]

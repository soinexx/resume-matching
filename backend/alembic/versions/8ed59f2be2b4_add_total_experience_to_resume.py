"""add_total_experience_to_resume

Revision ID: 8ed59f2be2b4
Revises: 010_add_unified_metrics
Create Date: 2026-02-10 12:28:16.898430

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8ed59f2be2b4'
down_revision: Union[str, None] = '010_add_unified_metrics'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add total_experience_months column to resumes table
    op.add_column('total_experience_months', sa.Integer(), nullable=True)


def downgrade() -> None:
    # Remove total_experience_months column from resumes table
    op.drop_column('total_experience_months')

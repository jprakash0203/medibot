"""Role-Based Access Control mapping."""

from enum import StrEnum


class Role(StrEnum):
    """User roles."""
    DOCTOR = "doctor"
    NURSE = "nurse"
    BILLING_EXECUTIVE = "billing_executive"
    TECHNICIAN = "technician"
    ADMIN = "admin"



class Collection(StrEnum):
    """ document collections."""
    GENERAL = "general"
    CLINICAL = "clinical"
    NURSING = "nursing"
    BILLING = "billing"
    EQUIPMENT = "equipment"

ROLE_COLLECTIONS: dict[Role, list[Collection]] = {
    Role.DOCTOR: [Collection.CLINICAL, Collection.NURSING, Collection.GENERAL],
    Role.NURSE: [Collection.NURSING, Collection.GENERAL],
    Role.BILLING_EXECUTIVE: [Collection.BILLING, Collection.GENERAL],
    Role.TECHNICIAN: [Collection.EQUIPMENT, Collection.GENERAL],
    Role.ADMIN: list(Collection),
}


def collections_for_role(role: Role) -> list[Collection]:
    """Return document collections accessible to the given role."""
    return ROLE_COLLECTIONS[role]


def roles_for_collection(collection: Collection) -> list[str]:
    """Return role names that may access the given collection (for chunk metadata)."""
    return [
        role.value
        for role, collections in ROLE_COLLECTIONS.items()
        if collection in collections
    ]


def can_use_sql_rag(role: Role) -> bool:
    """Only admins may query the operational SQLite database."""
    return role == Role.ADMIN

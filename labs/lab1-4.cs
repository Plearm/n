using System;
using System.Text.RegularExpressions;

public static class Validator
{

    public static bool ValidateName(string name)
    {
        return !string.IsNullOrWhiteSpace(name);
    }

    public static bool ValidateOwnershipType(string ownershipType)
    {
        return !string.IsNullOrWhiteSpace(ownershipType);
    }

    public static bool ValidateAddress(string address)
    {
        return !string.IsNullOrWhiteSpace(address);
    }

    public static bool ValidatePhone(string phone)
    {
        return Regex.IsMatch(phone, @"^\+7\d{10}$");
    }

    public static bool ValidateContactPerson(string contactPerson)
    {
        return !string.IsNullOrWhiteSpace(contactPerson);
    }
}

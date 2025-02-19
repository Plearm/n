using System;

public class Client
{
    private string _name;
    private string _ownershipType;
    private string _address;
    private string _phone;
    private string _contactPerson;

    public Client(string name, string ownershipType, string address, string phone, string contactPerson)
    {
        if (!Validator.ValidateName(name)) throw new ArgumentException("Некорректное название компании");
        if (!Validator.ValidateOwnershipType(ownershipType)) throw new ArgumentException("Некорректная форма собственности");
        if (!Validator.ValidateAddress(address)) throw new ArgumentException("Некорректный адрес");
        if (!Validator.ValidatePhone(phone)) throw new ArgumentException("Некорректный номер телефона");
        if (!Validator.ValidateContactPerson(contactPerson)) throw new ArgumentException("Некорректное контактное лицо");

        _name = name;
        _ownershipType = ownershipType;
        _address = address;
        _phone = phone;
        _contactPerson = contactPerson;
    }
}

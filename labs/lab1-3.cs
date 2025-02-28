using System;
using System.Text.RegularExpressions;

public class Client
{
    // Приватные поля
    private string _name;
    private string _ownershipType;
    private string _address;
    private string _phone;
    private string _contactPerson;

    // Конструктор
    public Client(string name, string ownershipType, string address, string phone, string contactPerson)
    {
        Name = name;  // Используем свойства, чтобы сразу проходила валидация
        OwnershipType = ownershipType;
        Address = address;
        Phone = phone;
        ContactPerson = contactPerson;
    }

    // Свойства с валидацией через Validator
    public string Name
    {
        get => _name;
        set
        {
            if (!Validator.ValidateName(value))
                throw new ArgumentException("Название компании не может быть пустым!");
            _name = value;
        }
    }

    public string OwnershipType
    {
        get => _ownershipType;
        set
        {
            if (!Validator.ValidateOwnershipType(value))
                throw new ArgumentException("Форма собственности не может быть пустой!");
            _ownershipType = value;
        }
    }

    public string Address
    {
        get => _address;
        set
        {
            if (!Validator.ValidateAddress(value))
                throw new ArgumentException("Адрес не может быть пустым!");
            _address = value;
        }
    }

    public string Phone
    {
        get => _phone;
        set
        {
            if (!Validator.ValidatePhone(value))
                throw new ArgumentException("Телефон должен быть в формате +7XXXXXXXXXX!");
            _phone = value;
        }
    }

    public string ContactPerson
    {
        get => _contactPerson;
        set
        {
            if (!Validator.ValidateContactPerson(value))
                throw new ArgumentException("Контактное лицо не может быть пустым!");
            _contactPerson = value;
        }
    }

    // Метод для отображения полной информации о клиенте
    public override string ToString()
    {
    return $"Компания: {Name}\nФорма собственности: {OwnershipType}\nАдрес: {Address}\nТелефон: {Phone}\nКонтактное лицо: {ContactPerson}";
    }
}


Так же заменить DisplayInfo на соответствующий метод, который выводит на экран и так же проитать по нему информацию

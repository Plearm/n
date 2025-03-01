using System;
using System.Text.Json;
using System.Text.RegularExpressions;

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

    
    public Client(string csvLine)
    {
        var parts = csvLine.Split(';');
        if (parts.Length != 5) throw new ArgumentException("Некорректный формат строки. Ожидается 5 параметров.");

        if (!Validator.ValidateName(parts[0])) throw new ArgumentException("Некорректное название компании");
        if (!Validator.ValidateOwnershipType(parts[1])) throw new ArgumentException("Некорректная форма собственности");
        if (!Validator.ValidateAddress(parts[2])) throw new ArgumentException("Некорректный адрес");
        if (!Validator.ValidatePhone(parts[3])) throw new ArgumentException("Некорректный номер телефона");
        if (!Validator.ValidateContactPerson(parts[4])) throw new ArgumentException("Некорректное контактное лицо");
        
        _name = parts[0];
        _ownershipType = parts[1];
        _address = parts[2];
        _phone = parts[3];
        _contactPerson = parts[4];
    }

    
    public Client(string json, bool isJson)
    {
        if (!isJson) throw new ArgumentException("Для создания из JSON необходимо передать 'true' вторым аргументом.");

        if (!Validator.ValidateName(obj.Name)) throw new ArgumentException("Некорректное название компании");
        if (!Validator.ValidateOwnershipType(obj.OwnershipType)) throw new ArgumentException("Некорректная форма собственности");
        if (!Validator.ValidateAddress(obj.Address)) throw new ArgumentException("Некорректный адрес");
        if (!Validator.ValidatePhone(obj.Phone)) throw new ArgumentException("Некорректный номер телефона");
        if (!Validator.ValidateContactPerson(obj.ContactPerson)) throw new ArgumentException("Некорректное контактное лицо");
        
        var obj = JsonSerializer.Deserialize<Client>(json);
        if (obj == null) throw new ArgumentException("Ошибка десериализации JSON.");

        _name = obj.Name;
        _ownershipType = obj.OwnershipType;
        _address = obj.Address;
        _phone = obj.Phone;
        _contactPerson = obj.ContactPerson;
    }

    
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

    
    public override bool Equals(object? obj)
    {
        if (obj is not Client other) return false;
        return Name == other.Name &&
               OwnershipType == other.OwnershipType &&
               Address == other.Address &&
               Phone == other.Phone &&
               ContactPerson == other.ContactPerson;
    }

    
    public override string ToString()
    {
        return $"Компания: {_name}, Телефон: {_phone}, Контактное лицо: {_contactPerson}";
    }
}


public static class Validator
{
    public static bool ValidateName(string name) => !string.IsNullOrWhiteSpace(name);
    public static bool ValidateOwnershipType(string ownershipType) => !string.IsNullOrWhiteSpace(ownershipType);
    public static bool ValidateAddress(string address) => !string.IsNullOrWhiteSpace(address);
    public static bool ValidatePhone(string phone) => Regex.IsMatch(phone, @"^\+7\d{10}$");
    public static bool ValidateContactPerson(string contactPerson) => !string.IsNullOrWhiteSpace(contactPerson);
}


public class ClientShort
{
    public string Name { get; }
    public string Phone { get; }
    public string ContactPerson { get; }

    public ClientShort(Client client)
    {
        Name = client.Name;
        Phone = client.Phone;
        ContactPerson = client.ContactPerson;
    }

    public override string ToString()
    {
        return $"Компания: {Name}, Телефон: {Phone}, Контактное лицо: {ContactPerson}";
    }
}


public class BusinessClient : Client
{
    public string INN { get; }
    public string OGRN { get; }

    public BusinessClient(string name, string ownershipType, string address, string phone, string contactPerson, string inn, string ogrn)
        : base(name, ownershipType, address, phone, contactPerson)
    {
        INN = inn;
        OGRN = ogrn;
    }

    public override bool Equals(object? obj)
    {
        if (!base.Equals(obj)) return false;
        if (obj is not BusinessClient other) return false;
        return INN == other.INN && OGRN == other.OGRN;
    }

    public override string ToString()
    {
        return base.ToString() + $"\nИНН: {INN}, ОГРН: {OGRN}";
    }
}


class Program
{
    static void Main()
    {
        Client client1 = new Client("ООО Ромашка", "ООО", "Москва", "+71234567890", "Матемрл Попрлр");
        Console.WriteLine(client);

        Client client2 = new Client("ООО Лилия;ООО;СПб; +79876543210;Петр Петров");
        
        string json = "{ \"Name\": \"ООО Лилия\", \"OwnershipType\": \"ООО\", \"Address\": \"СПб\", \"Phone\": \"+79876543210\", \"ContactPerson\": \"Петр Петров\" }";
        Client client3 = new Client(json, true);

        Client client4 = new Client("ООО Ромашка", "ООО", "Москва", "+71234567890", "Иван Иванов");
        Client client5 = new Client("ООО Ромашка", "ООО", "Москва", "+71234567890", "Иван Иванов");
        Console.WriteLine(client4.Equals(client5));
        
        ClientShort shortClient1 = new ClientShort(client);
        Console.WriteLine(shortClient);

        BusinessClient businessClient1 = new BusinessClient("ООО Лилия", "ООО", "СПб", "+79876543210", "Максим Лаптев", "1234567890", "0987654321");
        Console.WriteLine(businessClient);
    }
}

using System;
using System.Text.Json;
using System.Text.RegularExpressions;

// Лабораторная #1 - Класс Client
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

    public string Name => _name;
    public string OwnershipType => _ownershipType;
    public string Address => _address;
    public string Phone => _phone;
    public string ContactPerson => _contactPerson;

    public void DisplayFullInfo()
    {
        Console.WriteLine($"Название: {_name}\nВид собственности: {_ownershipType}\nАдрес: {_address}\nТелефон: {_phone}\nКонтактное лицо: {_contactPerson}");
    }

    public override string ToString()
    {
        return $"Компания: {_name}, Телефон: {_phone}, Контактное лицо: {_contactPerson}";
    }
}

// Лабораторная #2 - Класс Validator
public static class Validator
{
    public static bool ValidateName(string name) => !string.IsNullOrWhiteSpace(name);
    public static bool ValidateOwnershipType(string ownershipType) => !string.IsNullOrWhiteSpace(ownershipType);
    public static bool ValidateAddress(string address) => !string.IsNullOrWhiteSpace(address);
    public static bool ValidatePhone(string phone) => Regex.IsMatch(phone, @"^\+7\d{10}$");
    public static bool ValidateContactPerson(string contactPerson) => !string.IsNullOrWhiteSpace(contactPerson);
}

// Лабораторная #3 - Перегрузка конструкторов
// 🔹 **1. Обычный конструктор (ручной ввод данных)**
public Client(string name, string ownershipType, string address, string phone, string contactPerson)
    {
        Name = name;
        OwnershipType = ownershipType;
        Address = address;
        Phone = phone;
        ContactPerson = contactPerson;
    }

    // 🔹 **2. Конструктор из строки (CSV-формат)**
public Client(string csvLine)
    {
        var parts = csvLine.Split(';');
        if (parts.Length != 5) throw new ArgumentException("Некорректный формат строки. Ожидается 5 параметров.");

        Name = parts[0];
        OwnershipType = parts[1];
        Address = parts[2];
        Phone = parts[3];
        ContactPerson = parts[4];
    }

    // 🔹 **3. Конструктор из JSON (используем библиотеку Newtonsoft.Json)**
public Client(string json, bool isJson)
    {
        if (!isJson) throw new ArgumentException("Для создания из JSON необходимо передать 'true' вторым аргументом.");

        var obj = JsonConvert.DeserializeObject<Client>(json);
        if (obj == null) throw new ArgumentException("Ошибка десериализации JSON.");

        Name = obj.Name;
        OwnershipType = obj.OwnershipType;
        Address = obj.Address;
        Phone = obj.Phone;
        ContactPerson = obj.ContactPerson;
    }

// Лабораторная #4 - Устранение дублирования кода
// (Рефакторинг методов валидации в Validator)

// Лабораторная #5 - Оптимизация вывода данных
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

    public void DisplayShortInfo()
    {
        Console.WriteLine($"Компания: {Name}, Телефон: {Phone}, Контактное лицо: {ContactPerson}");
    }
}

// Лабораторная #7, #8 - Наследование и структуры данных
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

    public override void DisplayFullInfo()
    {
        base.DisplayFullInfo();
        Console.WriteLine($"ИНН: {INN}\nОГРН: {OGRN}");
    }
}

class Program
{
    static void Main()
    {
        // Тестирование
        Client client = new Client("ООО Ромашка", "ООО", "Москва, ул. Ленина, д.1", "+71234567890", "Иван Иванов");
        client.DisplayFullInfo();
        
        ClientShort shortClient = new ClientShort(client);
        shortClient.DisplayShortInfo();

        BusinessClient businessClient = new BusinessClient("ООО Лилия", "ООО", "СПб, ул. Кирова, д.5", "+79876543210", "Петр Петров", "1234567890", "0987654321");
        businessClient.DisplayFullInfo();
    }
}

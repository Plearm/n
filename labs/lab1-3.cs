using System;

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
        _name = name;
        _ownershipType = ownershipType;
        _address = address;
        _phone = phone;
        _contactPerson = contactPerson;
    }

    // Свойства (геттеры и сеттеры)
    public string Name 
    {
        get => _name;
        set => _name = value;
    }
Проверка через Set не работает (как я понял)
    public string OwnershipType
    {
        get => _ownershipType;
        set => _ownershipType = value;
    }

    public string Address
    {
        get => _address;
        set => _address = value;
    }

    public string Phone
    {
        get => _phone;
        set => _phone = value;
    }

    public string ContactPerson
    {
        get => _contactPerson;
        set => _contactPerson = value;
    }

    // Метод для отображения полной информации о клиенте
    public void DisplayInfo()
    {
        Console.WriteLine($"Компания: {Name}\nФорма собственности: {OwnershipType}\nАдрес: {Address}\nТелефон: {Phone}\nКонтактное лицо: {ContactPerson}");
    }
}


Так же заменить DisplayInfo на соответствующий метод, который выводит на экран и так же проитать по нему информацию

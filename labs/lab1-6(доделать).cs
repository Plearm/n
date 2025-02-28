using System;
using System.Text.Json;

public class Client
{
    // Поля класса (инкапсулированы)
    private string name;
    private string ownershipType;
    private string address;
    private string phone;
    private string contactPerson;

    // Основной конструктор
    public Client(string name, string ownershipType, string address, string phone, string contactPerson)
    {
        Validate(name, ownershipType, address, phone, contactPerson); 
        this.name = name;
        this.ownershipType = ownershipType;
        this.address = address;
        this.phone = phone;
        this.contactPerson = contactPerson;
    }

    // Перегруженный конструктор: строка (разделенная ;)
    public Client(string data)
    {
        string[] parts = data.Split(';');
        if (parts.Length != 5)
        {
            throw new ArgumentException("Некорректный формат строки. Должно быть 5 значений, разделенных ';'");
        }

        Validate(parts[0], parts[1], parts[2], parts[3], parts[4]);
        this.name = parts[0];
        this.ownershipType = parts[1];
        this.address = parts[2];
        this.phone = parts[3];
        this.contactPerson = parts[4];
    }
    

    public Client(JsonElement json)
    {
        try
        {
            this.name = json.GetProperty("name").GetString();
            this.ownershipType = json.GetProperty("ownershipType").GetString();
            this.address = json.GetProperty("address").GetString();
            this.phone = json.GetProperty("phone").GetString();
            this.contactPerson = json.GetProperty("contactPerson").GetString();

            Validate(name, ownershipType, address, phone, contactPerson); 
        }
        catch (Exception ex)
        {
            throw new ArgumentException("Ошибка обработки JSON: " + ex.Message);
        }
    }

    private void Validate(string name, string ownershipType, string address, string phone, string contactPerson)
    {
        if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(ownershipType) ||
            string.IsNullOrWhiteSpace(address) || string.IsNullOrWhiteSpace(phone) || string.IsNullOrWhiteSpace(contactPerson))
        {
            throw new ArgumentException("Все поля должны быть заполнены!");
        }
    }

    public override string ToString()
    {
        return $"Название: {name}\nВид собственности: {ownershipType}\nАдрес: {address}\nТелефон: {phone}\nКонтактное лицо: {contactPerson}";
    }
}

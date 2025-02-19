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

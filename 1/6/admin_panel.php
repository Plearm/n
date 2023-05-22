<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="admin_panel.css">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Задание 6</title>
</head>
<body>
<?php
$user = 'u52830';
$pass = '7841698';
$db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, [PDO::ATTR_PERSISTENT => true]);
if (!empty($messages)) {
    print('<div id="messages">');
    // Выводим все сообщения.
    foreach ($messages as $message) {
        print($message);
    }
    print('</div>');
}
$stmt = $db->prepare("SELECT count(idap) from userconnection where idsuper = 1;");
$stmt->execute();
$bessmert = $stmt->fetchColumn();
$bessmert = empty($bessmert) ? '0' : $bessmert;
    
$stmt = $db->prepare("SELECT count(idap) from userconnection where idsuper = 2;");
$stmt->execute();
$proh_skv_st = $stmt->fetchColumn();
$proh_skv_st = empty($proh_skv_st) ? '0' : $proh_skv_st;
    
$stmt = $db->prepare("SELECT count(idap) from userconnection where idsuper = 3;");
$stmt->execute();
$levitation = $stmt->fetchColumn();
$levitation = empty($levitation) ? '0' : $levitation;
    
print('<div class="sup_see">Бессмертие: ' . $bessmert . '<br>');
print('Прохождение сквозь стены: ' . $proh_skv_st . '<br>');
print('Левитация: ' . $levitation . '</div><br>');
?>

<form action="" method="POST">
    <table>
        <caption>Данные</caption>
        <tr>
            <th>id</th>
            <th>Имя</th>
            <th>email</th>
            <th>Год рождения</th>
            <th>Пол</th>
            <th>Кол-во конечностей</th>
            <th>Сверхспособности</th>
            <th>Биография</th>
        </tr>
        <?php
        foreach ($values as $value) {
            $a = $value['id'];
            echo '<tr><td>';
            print($a);
            echo '</td>';
            echo '<td>
                <input class="inp" name="fio' . $a . '" value="';
            print(htmlspecialchars($value['name']));
            echo '">
              </td>
              <td>
                <input class="inp" name="email' . $a . '" value="';
            print(htmlspecialchars($value['email']));
            echo '">
              </td>
              <td>
                <select name="year' . $a . '">';
            for ($i = 1922; $i < $value['year']; $i++) {
                printf('<option value="%d">%d год</option>', $i, $i);
            }
            printf('<option selected="selected" value="%d">%d год</option>', $i, $i);
            for ($i = $value['year'] + 1; $i < 2022; $i++) {
                printf('<option value="%d">%d год</option>', $i, $i);
            }
            echo '</select>
              </td><td>';
            if (htmlspecialchars($value['pol'] == 'M')) {
                printf('<label class="pot"><input type="radio" name="pol' . $a . '" value="M" checked="checked">M</label>');
                printf('<label class="pot"><input type="radio" name="pol' . $a . '" value="W">W</label>');
            } else {
                if (htmlspecialchars($value['pol'] == 'W')) {
                printf('<label class="pot"><input type="radio" name="pol' . $a . '" value="M">M</label>');
                printf('<label class="pot"><input type="radio" name="pol' . $a . '" value="W" checked="checked">W</label>');
                }
            }
            echo '</td><td>';
            for ($i = 1; $i < $value['kol_kon']; $i++) {
                printf('<input type="radio" name="limbs' . $a . '" value="%d"/>%d</label>', $i, $i);
            }
            printf('<input type="radio" name="limbs' . $a . '" checked="checked" value="%d">%d</label>', $value['kol_kon'], $value['kol_kon']);
            for ($i = $value['kol_kon'] + 1; $i <= 5; $i++) {
                printf('<input type="radio" name="limbs' . $a . '" value="%d"/>%d</label>', $i, $i);
            }
            echo '</td>';
            $mas = ['бессмертие', 'прохождение сквозь стены', 'левитация'];
            $flag = [0, 0, 0];
            $stmt = $db->prepare("SELECT idsuper FROM userconnection WHERE idap = ?");
            $stmt->execute([$a]);
            $abilities = $stmt->fetchAll(PDO::FETCH_COLUMN);
            printf('<td><select name="super' . $a . '[]" multiple="multiple">');
            foreach ($abilities as $sup) {
                if ($mas[(int)$sup - 1]) {
                    printf('<option value="%d" selected="selected">%s</option>', $sup, $mas[(int)$sup - 1]);
                    $flag[(int)$sup - 1] = 1;
                }
            }
            for ($i = 0; $i < sizeof($flag); $i++) {
                if (!$flag[$i]) {
                    printf('<option value="%d" >%s</option>', $i + 1, $mas[$i]);
                }
            }
            printf('</select></td>');
            echo '<td><textarea name="biography' . $a . '" id="" cols="20" rows="5" maxlength="256">';
            print htmlspecialchars($value['biography']);
            echo '</textarea></td>';
            echo '<td>
                    <input name="save' . $a . '" type="submit" value="save' . $a . '"/>
                    <input name="clear' . $a . '" type="submit" value="clear' . $a . '"/>
            </td>
        </tr>';
        }
        ?>
    </table>
   <?php echo '<input type="hidden" name="token" value="' . $_SESSION["token"] . '">'; ?>
</form>

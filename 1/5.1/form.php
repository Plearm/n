<html>
  <head>
    <link rel="stylesheet" href="form.css">
    <style>
/* Сообщения об ошибках и поля с ошибками выводим с красным бордюром. */
.error {
  border: 2px solid red;
}
    </style>
  </head>
  <body>

<?php
if (!empty($messages)) {
  print('<div id="messages">');
  // Выводим все сообщения.
  foreach ($messages as $message) {
    print($message);
  }
  print('</div>');
}

// Далее выводим форму отмечая элементы с ошибками классом error
// и задавая начальные значения элементов ранее сохраненными.
?>

<div class="formss">
    <form action="index.php" method="POST">
        <label>ФИО:<br/><input name="fio" <?php if ($errors['fio']) {print 'class="error"';} ?> value="<?php print $values['fio']; ?>"/></label>
        <label>Почта:<br/><input name="email" <?php if ($errors['email']) {print 'class="error"';} ?> value="<?php print $values['email']; ?>"/></label>
        <label>Год рождения:<br/>
            <?php
            if ($errors['year']) {
                print('<select name="year" class="error">');
                for ($i = 1922; $i <= 2022; $i++) {
                    printf('<option value="%d">%d год</option>', $i, $i);
                }
            }
            else {
                if($values['year']=='' || $values['year']<1922 || $values['year']>2022){
                    printf('<select name="year">');
                    for ($i = 1922; $i <= 2022; $i++) {
                        printf('<option value="%d">%d год</option>', $i, $i);
                    }
                }
                else {
                    printf('<select name="year">');
                    for ($i = 1922; $i < $values['year']; $i++) {
                        printf('<option value="%d">%d год</option>', $i, $i);
                    }
                    printf('<option value="%d" selected="selected">%d год</option>', $i, $i);
                    for ($i = $values['year'] + 1; $i <= 2022; $i++) {
                        printf('<option value="%d">%d год</option>', $i, $i);
                    }
                }
            }
            ?>
            </select>
        </label>
        Пол:<br/>
        <div class="poler">
        <?php
        if($values['pol'] == 'W'){
            printf('<label class="pot"><input type="radio" name="pol" value="M">M</label>');
            printf('<label class="pot"><input type="radio" name="pol" value="W" checked="checked">W</label>');
        }
        else
            if($values['pol'] == 'M'){
                printf('<label class="pot"><input type="radio" name="pol" value="M" checked="checked">M</label>');
                printf('<label class="pot"><input type="radio" name="pol" value="W">W</label>');
            }
            else{
                if($errors['pol']){
                    printf('<label class="pot error" ><input type="radio" name="pol" value="M">M</label>');
                    printf('<label class="pot error" ><input type="radio" name="pol" value="W">W</label>');
                }
                else{
                    printf('<label class="pot"><input type="radio" name="pol" value="M">M</label>');
                    printf('<label class="pot"><input type="radio" name="pol" value="W">W</label>');
                }
            }
        ?>
        </div>
        <div class="limber">
        Кол-во конечностей:<br/>
        <?php
        if($errors['limbs']){
            for ($i = 1; $i <= 5; $i++)
                printf('<label class="error"><input type="radio" name="limbs" value="%d"/>%d</label>', $i, $i);
        }
        else
            if($values['limbs']==''){
                for ($i = 1; $i <= 5; $i++)
                    printf('<label><input type="radio" name="limbs" value="%d"/>%d</label>', $i, $i);
            }
            else{
                for ($i = 1; $i < $values['limbs']; $i++)
                    printf('<label><input type="radio" name="limbs" value="%d"/>%d</label>', $i, $i);
                printf('<label><input type="radio" name="limbs" checked="checked" value="%d">%d</label>', $values['limbs'], $values['limbs']);
                for ($i = $values['limbs']+1; $i <= 5; $i++)
                    printf('<label><input type="radio" name="limbs" value="%d"/>%d</label>', $i, $i);
            }
        ?>
        </div>
        <label>
            Сверхспособности:<br/>
            <?php
            $mas = ['бессмертие', 'прохождение сквозь стены', 'левитация'];
            $flag = [0, 0, 0];
            if($errors['super']){
                printf('<select name="super[]" class="error" multiple="multiple">');
                printf('<option value="1">бессмертие</option>
                <option value="2">прохождение сквозь стены</option>
                <option value="3">левитация</option></select>');
            }
            else
                if($values['super']==''){
                    printf('<select name="super[]" multiple="multiple">');
                    printf('<option value="1">бессмертие</option>
                    <option value="2">прохождение сквозь стены</option>
                    <option value="3">левитация</option></select>');
                }
                else{
                    printf('<select name="super[]" multiple="multiple">');
                    foreach($values['super'] as $sup){
                        if($mas[(int)$sup-1]){
                            printf('<option value="%d" selected="selected">%s</option>',$sup, $mas[(int)$sup-1]);
                            $flag[(int)$sup-1] = 1;
                        }
                    }
                    for($i=0;$i<sizeof($flag);$i++){
                        if(!$flag[$i]){
                            printf('<option value="%d" >%s</option>',$i+1, $mas[$i]);
                        }
                    }
                }
            printf('</select>');
            ?>
        </label><br/>
        <label>
            Биография:<br/>
            <textarea name="biography" <?php if ($errors['biography']) {print 'class="error"';} ?> value="<?php print $values['biography'];?>"><?php print $values['biography'];?></textarea>
        </label><br/>
        <label class="ccheck" <?php if ($errors['check-1']) {print 'class="error"';} ?>>
            <input type="checkbox" <?php if ($errors['check-1']) {print 'class="error"';} else if($values['check-1']!='') {print 'checked="checked"';}?> name="check-1"/>
            с контрактом ознакомлен (а)
        </label><br/>
        <input type="submit" value="Отправить"/>
        <?php if (!empty($_SESSION['login'])) {echo '<input type="hidden" name="token" value="' . $_SESSION["token"] . '">'; } ?>
    </form>
</div>
  </body>
</html>

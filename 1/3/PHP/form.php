<form action ="" method = "POST">
	<label>
			Name:<br />
			<input name="field-name-1" />
		  </label><br />
	<label>
			email:<br />
			<input name="field-email"
			  type="email" />
		  </label><br />
	<label>
			Birth-date :<br />
			<input name="field-date"
			  type="date" />
		  </label><br />
	Sex: <label><input type="radio" checked="checked"
			name="radio-group-1" value = "male" />
			Male </label>
		  <label><input type="radio"
			name="radio-group-1" value = "female" />
			Female </label><br />
	Amoflimbs: <br/> <label><input type="radio" checked="checked"
			name="radio-group-2" value = "1" />
			1 </label><br/>
		  <label><input type="radio"
			name="radio-group-2" value = "2" />
			2 </label><br />
			<label><input type="radio"
			name="radio-group-2" value = "3" />
			3 </label><br />
			<label><input type="radio"
			name="radio-group-2" value = "4" />
			4 </label><br />
			<label><input type="radio"
			name="radio-group-2" value = "5" />
			5 </label><br />
	<label>
			Superpowers:
			<br/>
			<select name="field-name-4[]"
			  multiple="multiple">
			  <option value="immortality">Immortality</option>
			  <option value="walkthroughwalls" >Walk through walls</option>
			  <option value="levitation">levitation</option>
			</select>
		  </label><br />
	<label>
	Biography: <br/>
	  <textarea name = "bio-field"> </textarea>
	</label> <br/>
	<label>
	  <input type = "checkbox" name = "checkbox" /> </label> <br/>
	  Send: <br/>
	  <input type="submit" value="Sending" />
  </form>

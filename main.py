import argparse
import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import numpy as np
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_batch_size', type=int, default=1)
  parser.add_argument('--max_seq_len', type=int, default=16384)
  args = parser.parse_args()

  model_name = "Erland/Llama-3.2-3B-JAX"  # Replace with your repository name
  # tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = FlaxAutoModelForCausalLM.from_pretrained(
      model_name, from_pt=False) # from_pt should be False since it's already flax

  # Example prompt
  # prompt = "The quick brown fox"

  # Tokenize the prompt
  # prompt = tokenizer(prompt, return_tensors="np")
  input_ids = np.random.randint(
      0, 128000,
      (args.max_batch_size, args.max_seq_len // 2)
  )

  # Generate text
  output_ids = model.generate(
      input_ids,
      do_sample=False,
      max_new_tokens=args.max_seq_len // 2,
      min_new_tokens=args.max_seq_len // 2
  )

  # Decode the generated text
  # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)